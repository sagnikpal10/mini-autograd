import torch
import pytest
from autograd.tensor import Tensor

def test_with_requires_grad_False():
    a = Tensor([1, 2, 3, 4, 5])
    b = a.sum()
    with pytest.raises(AssertionError):
        b.backward()

    assert not a.grad
    assert not b.grad
    assert b.data == 15.0
    assert b.requires_grad == False
    assert len(b.parents) == 0

    a = torch.tensor([1, 2, 3, 4, 5])
    b = a.sum()
    with pytest.raises(RuntimeError):
        b.backward()

    assert not a.grad
    assert not b.grad
    assert b.data == 15.0
    assert b.requires_grad == False
    assert not b.grad_fn

def test_without_passing_gradient_tensor_0():
    a = Tensor(2.0, requires_grad=True)
    b = a.sum()
    b.backward()

    assert a.grad.data == 1.0
    assert b.grad.data == 1.0
    assert b.data == 2.0
    assert b.requires_grad == True
    assert len(b.parents) == 1

    a = torch.tensor(2.0, requires_grad=True)
    b = a.sum()
    b.retain_grad()
    b.backward()

    assert a.grad.data == 1.0
    assert b.grad.data == 1.0
    assert b.data == 2.0
    assert b.requires_grad == True
    assert len(b.grad_fn.next_functions) == 1


def test_without_passing_gradient_tensor_1():
    a = Tensor([1, 2, 3, 4, 5], requires_grad=True)
    b = a.sum()
    b.backward()

    assert list(a.grad.data) == [1, 1, 1, 1, 1]
    assert b.grad.data == 1.0
    assert b.data == 15.0
    assert b.requires_grad == True
    assert len(b.parents) == 1

    a = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
    b = a.sum()
    b.retain_grad()
    b.backward()

    assert list(a.grad.data) == [1, 1, 1, 1, 1]
    assert b.grad.data == 1.0
    assert b.data == 15.0
    assert b.requires_grad == True
    assert len(b.grad_fn.next_functions) == 1


def test_with_passing_gradient_tensor_0():
    a = Tensor([1, 2, 3, 4, 5], requires_grad=True)
    b = a.sum()
    b.backward(Tensor(7.0))

    assert list(a.grad.data) == [7, 7, 7, 7, 7]
    assert b.grad.data == 7.0
    assert b.data == 15.0
    assert b.requires_grad == True
    assert len(b.parents) == 1

    a = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
    b = a.sum()
    b.retain_grad()
    b.backward(torch.tensor(7.0))

    assert list(a.grad.data) == [7, 7, 7, 7, 7]
    assert b.grad.data == 7.0
    assert b.data == 15.0
    assert b.requires_grad == True
    assert len(b.grad_fn.next_functions) == 1

def test_with_passing_gradient_tensor_1():
    a = Tensor(2.0, requires_grad=True)
    b = a.sum()
    b.backward(Tensor(7.0))

    assert a.grad.data == 7.0
    assert b.grad.data == 7.0
    assert b.data == 2.0
    assert b.requires_grad == True
    assert len(b.parents) == 1

    a = torch.tensor(2.0, requires_grad=True)
    b = a.sum()
    b.retain_grad()
    b.backward(torch.tensor(7.0))

    assert a.grad.data == 7.0
    assert b.grad.data == 7.0
    assert b.data == 2.0
    assert b.requires_grad == True
    assert len(b.grad_fn.next_functions) == 1