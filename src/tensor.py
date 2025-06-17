import numpy as np


class Tensor:
    def __init__(self, data, requires_grad=True, parent=None):
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(data) if requires_grad else None
        self.parent = parent

    def set_parent(self, p):
        self.parent = p

    def backward(self, grad=None):
        if not self.requires_grad: return None

        if grad is None:
            grad = np.ones_like(self.data) # initialize with 1 because dL/dL = 1

        self.grad += grad

        if self.parent is not None:
            grads, inputs = self.parent.backward(self.grad) # call all parents' backward
            for grad, input in zip(grads, inputs):
                input.backward(grad)

    def zero_grad(self):
        self.grad.fill(0.0)

    def __repr__(self):
        return f"Tensor(data:{self.data}\ngrad:{self.grad})"       

