import torch
from autograd.tensor import Tensor
from autograd.functions import exp, pow, sqrt, log, tanh

def test_0():
    a = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    b = Tensor([[10, 20, 30], [41, 51, 61]], requires_grad=True)
    c = a + b
    d = Tensor([[1, 2, 1], [1, 1, 2]], requires_grad=True)
    e = c * d
    f = Tensor([[-1.0, 1.0], [2.0, -1.0], [-1.9, 1.0]], requires_grad=True)
    h = e @ f
    h.backward(Tensor([[-1.0, 1.0], [2.0, -1.0]]))

    a_ = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    b_ = torch.tensor([[10.0, 20.0, 30.0], [41.0, 51.0, 61.0]], requires_grad=True)
    c_ = a_ + b_
    d_ = torch.tensor([[1.0, 2.0, 1.0], [1.0, 1.0, 2.0]], requires_grad=True)
    e_ = c_ * d_
    f_ = torch.tensor([[-1.0, 1.0], [2.0, -1.0], [-1.9, 1.0]], requires_grad=True)
    h_ = e_ @ f_
    h_.backward(torch.tensor([[-1.0, 1.0], [2.0, -1.0]]))
    
    assert [[round(float(x), 3) for x in row] for row in a.grad.data.tolist()] == [[round(float(x), 3) for x in row] for row in a_.grad.data.tolist()]
    assert [[round(float(x), 3) for x in row] for row in b.grad.data.tolist()] == [[round(float(x), 3) for x in row] for row in b_.grad.data.tolist()]
    assert [[round(float(x), 3) for x in row] for row in d.grad.data.tolist()] == [[round(float(x), 3) for x in row] for row in d_.grad.data.tolist()]
    assert [[round(float(x), 3) for x in row] for row in f.grad.data.tolist()] == [[round(float(x), 3) for x in row] for row in f_.grad.data.tolist()]


def test_1():
    a = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    b = Tensor([[10, 20, 30], [41, 51, 61]], requires_grad=True)
    c = a + b
    d = Tensor([[1, 2, 1], [1, 1, 2]], requires_grad=True)
    e = c * d
    f = Tensor([[-1.0, 1.0], [2.0, -1.0], [-1.9, 1.0]], requires_grad=True)
    h = e @ f
    g = h.sum()
    g.backward()

    a_ = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    b_ = torch.tensor([[10.0, 20.0, 30.0], [41.0, 51.0, 61.0]], requires_grad=True)
    c_ = a_ + b_
    d_ = torch.tensor([[1.0, 2.0, 1.0], [1.0, 1.0, 2.0]], requires_grad=True)
    e_ = c_ * d_
    f_ = torch.tensor([[-1.0, 1.0], [2.0, -1.0], [-1.9, 1.0]], requires_grad=True)
    h_ = e_ @ f_
    g_ = h_.sum()
    g_.backward()

    assert [[round(float(x), 3) for x in row] for row in a.grad.data.tolist()] == [[round(float(x), 3) for x in row] for row in a_.grad.data.tolist()]
    assert [[round(float(x), 3) for x in row] for row in b.grad.data.tolist()] == [[round(float(x), 3) for x in row] for row in b_.grad.data.tolist()]
    assert [[round(float(x), 3) for x in row] for row in d.grad.data.tolist()] == [[round(float(x), 3) for x in row] for row in d_.grad.data.tolist()]
    assert [[round(float(x), 3) for x in row] for row in f.grad.data.tolist()] == [[round(float(x), 3) for x in row] for row in f_.grad.data.tolist()]


def test_2():
    x = Tensor([[1, 2], [3, 4]], requires_grad=True)
    y = exp(x)
    z = tanh(y)
    w = z.sum()
    w.backward()

    x_ = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y_ = torch.exp(x_)
    z_ = torch.tanh(y_)
    w_ = z_.sum()
    w_.backward()

    assert [[round(float(x), 6) for x in row] for row in x.grad.data.tolist()] == [[round(float(x), 6) for x in row] for row in x_.grad.tolist()]