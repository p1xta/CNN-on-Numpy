import numpy as np

from src.module import Module


class Sequential(Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def params(self):
        parameters = []
        for layer in self.layers:
            if hasattr(layer, 'params'):
                parameters.extend(layer.params())
        return parameters
    
    def zero_grad(self):
        for param in self.params():
            param.zero_grad()

