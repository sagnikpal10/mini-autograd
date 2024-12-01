from autograd.tensor import Tensor

def test_add_0():
    a = Tensor([10, 12, 13], requires_grad=True)
    b = Tensor([10, 8, 7], requires_grad=True)
    c = a + b
    assert list(c.data) == [20, 20, 20]

    c.backward(Tensor([-1.0, -4.0, -7.0]))
    assert list(a.grad.data) == [-1.0, -4.0, -7.0]
    assert list(b.grad.data) == [-1.0, -4.0, -7.0]

def test_add_1():
    a = Tensor([[10, 20, 30], [40, 50, 60]], requires_grad = True)
    b = Tensor([7, 8, 9], requires_grad = True)
    c = a + b
    assert c.data.tolist() == [[17, 28, 39], [47, 58, 69]]

    c.backward(Tensor([[1, -1, 1], [1, -1, 1]]))
    assert a.grad.data.tolist() == [[1, -1, 1], [1, -1, 1]]
    assert b.grad.data.tolist() == [2, -2, 2]

def test_add_2():
    a = Tensor([[10, 20, 30], [40, 50, 60]], requires_grad = True)
    b = Tensor([[7, 8, 9]], requires_grad = True)
    c = a + b
    assert c.data.tolist() == [[17, 28, 39], [47, 58, 69]]

    c.backward(Tensor([[1, -1, 1], [1, -1, 1]]))
    assert a.grad.data.tolist() == [[1, -1, 1], [1, -1, 1]]
    assert b.grad.data.tolist() == [[2, -2, 2]]