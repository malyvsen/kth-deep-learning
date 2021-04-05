import numpy as np
from crater import Tensor


def test_int():
    assert -Tensor.from_builtin(3) == Tensor.from_builtin(-3)


def test_float():
    assert -Tensor.from_builtin(2.5) == Tensor.from_builtin(-2.5)


def test_list():
    assert -Tensor.from_builtin([-1, 2.5]) == Tensor.from_builtin([1, -2.5])


def test_gradients():
    tensor = Tensor.from_builtin([2, 3, 4])
    result = -tensor
    gradients = result.backward(np.array([1, 2, 3]))
    assert np.all(gradients[tensor] == [-1, -2, -3])
