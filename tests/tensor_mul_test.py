from autograd.tensor import Tensor

def test_mul_0():
    a = Tensor([10, 2, 3], requires_grad=True)
    b = Tensor([2, 7, 9], requires_grad=True)
    c = a * b
    assert c.data.tolist() == [20.0, 14.0, 27.0]

    c.backward(Tensor([-1., -2., -3.]))
    assert a.grad.data.tolist() == [-2.0, -14.0, -27.0]
    assert b.grad.data.tolist() == [-10.0, -4.0, -9.0]

def test_mul_1():
    a = Tensor([[1, 21, 3], [4, 1, 6]], requires_grad = True)
    b = Tensor([7, 18, 5], requires_grad = True)
    c = a * b
    assert c.data.tolist() == [[7.0, 378.0, 15.0], [28.0, 18.0, 30.0]]

    c.backward(Tensor([[1, 1, 1], [1, 1, 1]]))
    assert a.grad.data.tolist() == [[7.0, 18.0, 5.0], [7.0, 18.0, 5.0]]
    assert b.grad.data.tolist() == [5.0, 22.0, 9.0]

def test_mul_2():
    a = Tensor([[1, 1, 3], [4, 7, 6]], requires_grad = True)
    b = Tensor([[7, 2, 11]], requires_grad = True)
    c = a * b
    assert c.data.tolist() == [[7.0, 2.0, 33.0], [28.0, 14.0, 66.0]]
    
    c.backward(Tensor([[1, 1, 1], [1, 1, 1]]))
    assert a.grad.data.tolist() == [[7.0, 2.0, 11.0], [7.0, 2.0, 11.0]]
    assert b.grad.data.tolist() == [[5.0, 8.0, 9.0]]