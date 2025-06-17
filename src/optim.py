import numpy as np


class Optimizer:
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr
    
    def zero_grad(self):
        for param in self.params:
            param.zero_grad()

    def step():
        raise NotImplementedError("Subclasses must implement method step()")

class SGD(Optimizer):
    def step(self):
        for param in self.params:
            param.data -= self.lr * param.grad


class SGDMomentum(Optimizer):
    def __init__(self, params, lr, momentum=0.9, weight_decay=0):
        super().__init__(params, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.v = [np.zeros_like(param.data) for param in self.params]

    def step(self):
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            
            grad = param.grad + self.weight_decay * param.data
            self.v[i] = self.momentum * self.v[i] - self.lr * grad            
            param.data += self.v[i]
