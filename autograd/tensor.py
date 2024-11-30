import numpy as np
from dataclasses import dataclass
from typing import Union

@dataclass
class GradDependency:
    tensor: 'Tensor'
    gradient_function: callable

class Tensor(object):

    def __init__(
            self, 
            data: Union[int, float, list, np.ndarray],
            parents: list[GradDependency] = None,
            requires_grad: bool = False
            ) -> None:
        assert isinstance(data, (int, float, list, np.ndarray)), "init arg 'data' must be one of these types [int, float, list, np.ndarray]"
        
        self.data = data if isinstance(data, np.ndarray) and data.dtype == np.float64 else np.array(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.parents = parents if parents else []
        self.shape = self.data.shape
        self.grad = Tensor(np.zeros_like(self.data, dtype=np.float64)) if self.requires_grad else None

    def __repr__(self) -> str:
        return f"<Tensor (data={self.data}, requires_grad={self.requires_grad}, shape={self.shape}), parents={self.parents}, grad={self.grad}>"
    
    def backward(self, grad_tensor: 'Tensor' = None):
        '''backward pass for calculating gradients using back propagation'''
        assert self.requires_grad, "Cannot call backward on a tensor with requires_grad=False"
        assert grad_tensor or self.shape == (), "grad tensor must be not-None or have 0-dim"
        
        if not grad_tensor and self.shape == ():
            grad_tensor = Tensor(1.0)

        self.grad.data += grad_tensor.data # "accumulating the gradient passed from child tensor"
        
        for parent in self.parents:
            backward_grad = parent.gradient_function(grad_tensor)
            parent.tensor.backward(backward_grad)
    
    def reset_grads_to_zero(self) -> None:
        '''setting the gradients of this tensor and all its child tensors to zero.'''
        if not self.requires_grad: return 
        self.grad = Tensor(np.zeros_like(self.data, dtype=np.float64))

        for tensor in self.parents:
            tensor.zero_grad()

    # Defining all tensor operations
    def sum(self) -> 'Tensor':
        '''returns (0-dim tensor) sum of all elements of a tensor (including every dim)'''
        child_tensor = Tensor(
            data=self.data.sum(),
            requires_grad = self.requires_grad,
        )
        if self.requires_grad:
            def grad_function(grad_tensor: 'Tensor') -> 'Tensor':
                return Tensor(grad_tensor.data * np.ones_like(self.data))
            child_tensor.parents = [GradDependency(tensor=self, gradient_function=grad_function)]

        return child_tensor