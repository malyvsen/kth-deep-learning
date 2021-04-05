import numpy as np
from crater import Tensor


def test_full():
    assert Tensor.from_builtin([1, 2, -1]).sum() == Tensor.from_builtin(2)


def test_specific():
    assert Tensor.from_builtin([[1, 4], [5, 0]]).sum(0) == Tensor.from_builtin([6, 4])
    assert Tensor.from_builtin([[1, 4], [5, 0]]).sum(1) == Tensor.from_builtin([5, 5])


def test_negative():
    assert Tensor.from_builtin([[1, 4], [5, 0]]).sum(-2) == Tensor.from_builtin([6, 4])
    assert Tensor.from_builtin([[1, 4], [5, 0]]).sum(-1) == Tensor.from_builtin([5, 5])


def test_gradients_full():
    tensor = Tensor.from_builtin([2, 3, 4])
    result = tensor.sum()
    gradients = result.backward(np.array(-1))
    assert np.all(gradients[tensor] == [-1, -1, -1])


def test_gradients_specific():
    tensor = Tensor.from_builtin([[5, 2], [9, 1]])
    result = tensor.sum(1)
    gradients = result.backward(np.array([1, 3]))
    assert np.all(gradients[tensor] == [[1, 1], [3, 3]])


def test_gradients_negative():
    tensor = Tensor.from_builtin([[5, 2], [9, 1]])
    result = tensor.sum(-1)
    gradients = result.backward(np.array([1, 3]))
    assert np.all(gradients[tensor] == [[1, 1], [3, 3]])
