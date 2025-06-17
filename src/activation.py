import numpy as np

from src.tensor import Tensor

class ReLU:
    def forward(self, x):
        self.input = x
        output = Tensor(np.maximum(0, x.data), requires_grad=x.requires_grad, parent=self)
        return output

    def backward(self, grad):
        return [grad * (self.input.data > 0).astype(np.float32)], [self.input]
    
    def __call__(self, x):
        return self.forward(x)
        