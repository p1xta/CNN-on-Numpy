# Deep Learning framework written only using NumPy.

It supports automatic differentiation, convolutional and linear layers, activations, optimizers, loss, and evaluation metrics. 

## Components:
```tensor.py``` - Tensor class with autograd engine
```loss.py``` - CrossEntropyLoss
```optim.py``` - SGD, SGD with Momentum
```activation.py``` - ReLU 
```model.py``` - Sequential model container
```module.py``` - Conv2d, Linear, ResidualBlock, Flatten layers
```metrics.py``` Accuracy, Precision, Recall