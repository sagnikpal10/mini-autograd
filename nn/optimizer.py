from nn.module import Module
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List

class SGD:
    def __init__(self, lr: float = 0.01) -> None:
        self.lr = lr

    def step(self, module: Module) -> None:
        for parameter in module.parameters():
            parameter.data = parameter.data - parameter.grad.data * self.lr
    

