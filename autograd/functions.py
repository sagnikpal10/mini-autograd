from autograd.tensor import GradDependency, Tensor
import numpy as np

def pow(parent: Tensor, power):
    assert type(power) == int
    child_tensor = Tensor(
        data=parent.data ** power,
        requires_grad = parent.requires_grad,
    )
    if parent.requires_grad:
        def grad_function(grad_tensor: 'Tensor') -> 'Tensor':
            return Tensor(grad_tensor.data * power * parent.data ** (power - 1))
        child_tensor.grad_fn = [GradDependency(tensor=parent, gradient_function=grad_function)]

    return child_tensor


def sqrt(parent: Tensor):
    child_tensor = Tensor(
        data=np.sqrt(parent.data),
        requires_grad = parent.requires_grad,
    )
    if parent.requires_grad:
        def grad_function(grad_tensor: 'Tensor') -> 'Tensor':
            return Tensor(- 1 / (2 * np.sqrt(parent.data)) * grad_tensor.data)
        child_tensor.grad_fn = [GradDependency(tensor=parent, gradient_function=grad_function)]

    return child_tensor


def exp(parent: Tensor):
    child_tensor = Tensor(
        data=np.exp(parent.data),
        requires_grad = parent.requires_grad,
    )
    if parent.requires_grad:
        def grad_function(grad_tensor: 'Tensor') -> 'Tensor':
            return Tensor(grad_tensor.data * parent.data)
        child_tensor.grad_fn = [GradDependency(tensor=parent, gradient_function=grad_function)]

    return child_tensor


def log(parent: Tensor):
    child_tensor = Tensor(
        data=np.log(parent.data),
        requires_grad = parent.requires_grad,
    )  
    if parent.requires_grad:
        def grad_function(grad_tensor: 'Tensor') -> 'Tensor':
            return Tensor(grad_tensor.data * np.divide(1, parent.data))
        child_tensor.grad_fn = [GradDependency(tensor=parent, gradient_function=grad_function)]

    return child_tensor

def tanh(parent: Tensor):
    child_tensor = Tensor(
        data=np.tanh(parent.data),
        requires_grad = parent.requires_grad,
    )
    if parent.requires_grad:
        def grad_function(grad_tensor: 'Tensor') -> 'Tensor':
            return Tensor(grad_tensor.data * (1 - np.tanh(parent.data) * np.tanh(parent.data)))
        child_tensor.grad_fn = [GradDependency(tensor=parent, gradient_function=grad_function)]

    return child_tensor