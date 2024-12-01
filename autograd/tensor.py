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
            grad_fn: list[GradDependency] = None,
            requires_grad: bool = False
            ) -> None:
        assert isinstance(data, (int, float, list, np.ndarray)), "init arg 'data' must be one of these types [int, float, list, np.ndarray]"
        
        self.data = data if isinstance(data, np.ndarray) and data.dtype == np.float64 else np.array(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad_fn = grad_fn if grad_fn else []
        self.shape = self.data.shape
        self.grad = Tensor(np.zeros_like(self.data, dtype=np.float64)) if self.requires_grad else None

    def __repr__(self) -> str:
        return f"<Tensor (data={self.data}, requires_grad={self.requires_grad}, shape={self.shape}), grad_fn={self.grad_fn}, grad={self.grad}>"
    
    def backward(self, grad_tensor: 'Tensor' = None):
        '''backward pass for calculating gradients using back propagation'''
        assert self.requires_grad, "Cannot call backward on a tensor with requires_grad=False"
        assert grad_tensor or self.shape == (), "grad tensor must be not-None or have 0-dim"
        
        if not grad_tensor and self.shape == ():
            grad_tensor = Tensor(1.0)

        self.grad.data += grad_tensor.data # "accumulating the gradient passed from child tensor"
        
        for parent in self.grad_fn:
            backward_grad = parent.gradient_function(grad_tensor)
            parent.tensor.backward(backward_grad)
    
    def zero_grad(self) -> None:
        '''setting the gradients of this tensor and all its child tensors to zero.'''
        if not self.requires_grad: return 
        self.grad = Tensor(np.zeros_like(self.data, dtype=np.float64))

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
            child_tensor.grad_fn = [GradDependency(tensor=self, gradient_function=grad_function)]

        return child_tensor
    
    def __add__(self, other: 'Tensor') -> 'Tensor':
        """returns element wise self + other"""
        grad_functions = []

        if self.requires_grad:
            def grad_function_self(grad: 'Tensor') -> 'Tensor':
                grad_tensor = Tensor(grad.data)
                ndims_added = grad_tensor.data.ndim - self.data.ndim
                for _ in range(ndims_added):
                    grad_tensor.data = grad_tensor.data.sum(axis=0)
                for i, dim in enumerate(self.shape):
                    if dim == 1:
                        grad_tensor.data = grad_tensor.data.sum(axis=i, keepdims=True)
                return grad_tensor
            grad_functions.append(GradDependency(tensor=self, gradient_function=grad_function_self))

        if other.requires_grad:
            def grad_function_other(grad: 'Tensor') -> 'Tensor':
                grad_tensor = Tensor(grad.data)
                ndims_added = grad_tensor.data.ndim - other.data.ndim
                for _ in range(ndims_added):
                    grad_tensor.data = grad_tensor.data.sum(axis=0)
                for i, dim in enumerate(other.shape):
                    if dim == 1:
                        grad_tensor.data = grad_tensor.data.sum(axis=i, keepdims=True)
                return grad_tensor
            grad_functions.append(GradDependency(tensor=other, gradient_function=grad_function_other))
        
        return Tensor(
            data=self.data + other.data,
            requires_grad = self.requires_grad or other.requires_grad,
            grad_fn = grad_functions
        )
    
    def __radd__(self, other: 'Tensor') -> 'Tensor':
        return self + other

    def __mul__(self, other) -> 'Tensor':
        """returns element wise multiplication of tensor"""
        grad_functions = []
        if self.requires_grad:
            def grad_function_self(grad: 'Tensor') -> 'Tensor':
                grad_tensor = Tensor(grad.data * other.data)
                ndims_added = grad_tensor.data.ndim - self.data.ndim
                for _ in range(ndims_added):
                    grad_tensor.data = grad_tensor.data.sum(axis=0)
                for i, dim in enumerate(self.shape):
                    if dim == 1:
                        grad_tensor.data = grad_tensor.data.sum(axis=i, keepdims=True)
                return grad_tensor
            grad_functions.append(GradDependency(tensor=self, gradient_function=grad_function_self))

        if other.requires_grad:
            def grad_function_other(grad: 'Tensor') -> 'Tensor':
                grad_tensor = Tensor(grad.data * self.data)
                ndims_added = grad_tensor.data.ndim - other.data.ndim
                for _ in range(ndims_added):
                    grad_tensor.data = grad_tensor.data.sum(axis=0)
                for i, dim in enumerate(other.shape):
                    if dim == 1:
                        grad_tensor.data = grad_tensor.data.sum(axis=i, keepdims=True)
                return grad_tensor
            grad_functions.append(GradDependency(tensor=other, gradient_function=grad_function_other))

        return Tensor(
            data=self.data * other.data,
            requires_grad = self.requires_grad or other.requires_grad,
            grad_fn = grad_functions
        )

    def __rmul__(self, other) -> 'Tensor':
        return other * self
    
    def _neg(self) -> 'Tensor':
        child_tensor = Tensor(
            data=-self.data,
            requires_grad = self.requires_grad,
        )
        if self.requires_grad:
            def grad_function(grad_tensor: 'Tensor') -> 'Tensor':
                return Tensor(-grad_tensor.data)
            child_tensor.grad_fn = [GradDependency(tensor=self, gradient_function=grad_function)]
        return child_tensor

    def __sub__(self, other) -> 'Tensor':
        return self + -other

    def __rsub__(self, other) -> 'Tensor':
        return other + -self

    def __matmul__(self, other) -> 'Tensor':
        grad_functions = []
        if self.requires_grad:
            def grad_function_self(grad_tensor: 'Tensor') -> 'Tensor':
                return Tensor(grad_tensor.data @ other.data.T)
            grad_functions.append(GradDependency(tensor=self, gradient_function=grad_function_self))

        if other.requires_grad:
            def grad_function_other(grad_tensor: 'Tensor') -> 'Tensor':
                return Tensor(self.data.T @ grad_tensor.data)
            grad_functions.append(GradDependency(tensor=other, gradient_function=grad_function_other))

        return Tensor(data=self.data @ other.data,
                    requires_grad=self.requires_grad or other.requires_grad,
                    grad_fn=grad_functions)


class Parameter(Tensor):
    def __init__(self, data = None, *shape) -> None:
        data = np.random.randn(*shape) if not data else data
        super().__init__(data, requires_grad=True)