import numpy as np

from src.tensor import Tensor
from src.activation import ReLU

np.random.seed = 42

class Module:
    def __init__(self):
        self._modules = []

    def __call__(self, *args):
        return self.forward(*args)
    
    def forward(self, *args):
        raise NotImplementedError("Subclasses must implement method forward()")
    
    def backward(self, grad):
        raise NotImplementedError("Subclasses must implement method backward()")
    
    def params(self):
        return []

class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = Tensor(np.random.randn(in_features, out_features) * np.sqrt(2 / in_features), requires_grad=True)
        self.b = Tensor(np.zeros(out_features), requires_grad=True)

    def forward(self, x:Tensor):
        self.input = x
        output = Tensor(x.data @ self.W.data + self.b.data, requires_grad=x.requires_grad)
        output.set_parent(self)

        return output

    def backward(self, grad):
        x = self.input

        dW = x.data.T @ grad
        db = np.sum(grad, axis=0)
        dx = grad @ self.W.data.T

        self.W.grad += dW
        self.b.grad += db

        return [dx], [x]

    def params(self):
        return [self.W, self.b]


class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding

        weight_shape = (self.out_channels, self.in_channels, *self.kernel_size)
        self.W = Tensor(np.random.randn(*weight_shape) * np.sqrt(1 / np.prod(weight_shape[1:])), requires_grad=True)
        self.b = Tensor(np.zeros(self.out_channels, dtype=np.float32), requires_grad=True)

    def _im2row(self, x):
        B, C, H, W = x.shape
        k_h, k_w = self.kernel_size
        H_out = (H + 2 * self.padding - k_h) // self.stride + 1
        W_out = (W + 2 * self.padding - k_w) // self.stride + 1

        x_padded = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')

        rows = []
        for i in range(0, H_out * self.stride, self.stride):
            for j in range(0, W_out * self.stride, self.stride):
                patch = x_padded[:, :, i:i + k_h, j:j + k_w]
                rows.append(patch.reshape(x.shape[0], -1))  # (B, C*k_h*k_w)
        return np.stack(rows, axis=1)  # (B, H_out*W_out, C*k_h*k_w)

    def _row2im(self, rows, x_shape, out_shape):
        B, C, H, W = x_shape
        k_h, k_w = self.kernel_size
        H_out, W_out = out_shape
        dx_padded = np.zeros((B, C, H + 2 * self.padding, W + 2 * self.padding))

        row_idx = 0
        for i in range(0, H_out * self.stride, self.stride):
            for j in range(0, W_out * self.stride, self.stride):
                patch = rows[:, row_idx, :].reshape(B, C, k_h, k_w)
                dx_padded[:, :, i:i + k_h, j:j + k_w] += patch
                row_idx += 1

        if self.padding > 0:
            return dx_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        return dx_padded
    
    def forward(self, x):
        self.input = x
        B, C, H, W = x.data.shape
        k_h, k_w = self.kernel_size

        self.X_rows = self._im2row(x.data)  # (B, num_patches, C*k_h*k_w)
        W_row = self.W.data.reshape(self.out_channels, -1)  # (out_ch, C*k_h*k_w)

        out = self.X_rows @ W_row.T + self.b.data  # (B, num_patches, out_ch)
        H_out = (H + 2 * self.padding - k_h) // self.stride + 1
        W_out = (W + 2 * self.padding - k_w) // self.stride + 1

        out = out.transpose(0, 2, 1).reshape(B, self.out_channels, H_out, W_out)
        out_tensor = Tensor(out, requires_grad=x.requires_grad)
        out_tensor.set_parent(self)
        return out_tensor

    def backward(self, grad):
        B, C, H, W = self.input.data.shape
        k_h, k_w = self.kernel_size
        H_out = (H + 2 * self.padding - k_h) // self.stride + 1
        W_out = (W + 2 * self.padding - k_w) // self.stride + 1

        grad_reshaped = grad.transpose(0, 2, 3, 1).reshape(B, -1, self.out_channels)  # transpose - (B, H_out, W_out, out_ch), reshape - (B, num_patches, out_ch)

        dW = np.zeros_like(self.W.data)
        db = np.zeros_like(self.b.data)
        dX_rows = np.zeros_like(self.X_rows)

        for b in range(B):
            print(grad_reshaped[b].shape)
            print(self.X_rows[b].shape)

            # (out_ch, num_patches) @ (num_patches, C*k_h*k_w) -> (out_ch, C*k_h*k_w) -> reshape
            dW += (grad_reshaped[b].T @ self.X_rows[b]).reshape(self.W.data.shape)
            db += np.sum(grad_reshaped[b], axis=0) # bias grad is 1, so we just add the sum of grad that came from input 
            # (num_patches, out_ch) @ (out_ch, C*k_h*k_w) -> (num_patches, C*k_h*k_w)
            dX_rows[b] = grad_reshaped[b] @ self.W.data.reshape(self.out_channels, -1)

        dW = dW.reshape(self.W.data.shape)

        dx = self._row2im(dX_rows, self.input.data.shape, (H_out, W_out))

        self.W.grad += dW
        self.b.grad += db

        return [dx], [self.input]

    def params(self):
        return [self.W, self.b]

class Add(Module):
    def forward(self, x, y):
        self.x = x
        self.y = y
        output = Tensor(x.data + y.data, requires_grad=x.requires_grad or y.requires_grad)
        output.set_parent(self)
        return output

    def backward(self, grad):
        return [grad, grad], [self.x, self.y] # d add / dx = d add / d y = 1


class ResidualBlock(Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu1 = ReLU()
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding)
        self.add = Add()
        self.relu2 = ReLU()

    def forward(self, x):
        self.input = x
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        
        identity = x
        
        out = self.add(out, identity)
        out = self.relu2(out)
        
        return out

    def backward(self, grad):
        grad, _ = self.relu2.backward(grad)
        
        grad_main, grad_skip = self.add.backward(grad)[0]
        
        grad_main, _ = self.conv2.backward(grad_main[0])
        grad_main, _ = self.relu1.backward(grad_main[0])
        grad_main, _ = self.conv1.backward(grad_main[0])
        
        total_grad = grad_main[0] + grad_skip
        
        return [total_grad], [self.input]

    def params(self):
        p = self.conv1.params() + self.conv2.params()
        return p
    
class Flatten(Module):
    def forward(self, x):
        self.input = x
        batch_size = x.data.shape[0]
        output = Tensor(x.data.reshape(batch_size, -1), requires_grad=x.requires_grad)
        output.set_parent(self)
        return output
    
    def backward(self, grad):
        dx = grad.reshape(self.input.data.shape)
        return [dx], [self.input]