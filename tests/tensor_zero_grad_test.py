from autograd.tensor import Tensor

def test_zero_grad_0():
    a = Tensor([2, 4, 6], requires_grad=True)
    b = Tensor([4, 5, 6], requires_grad=True)
    c = a * b
    assert c.data.tolist() == [8.0, 20.0, 36.0]

    c.backward(Tensor([-5.0, -2.0, -3.0]))
    assert a.grad.data.tolist() == [-20.0, -10.0, -18.0]
    assert b.grad.data.tolist() == [-10.0, -8.0, -18.0]

    a.zero_grad()
    b.zero_grad()
    assert a.grad.data.tolist() == [0.0, 0.0, 0.0]
    assert b.grad.data.tolist() == [0.0, 0.0, 0.0]
