import numpy as np
from crater import Tensor


def test_int():
    assert Tensor.from_builtin(2) / Tensor.from_builtin(4) == Tensor.from_builtin(0.5)


def test_float():
    assert Tensor.from_builtin(2.5) / Tensor.from_builtin(0.5) == Tensor.from_builtin(5)


def test_list():
    assert Tensor.from_builtin([1, 2.5]) / Tensor.from_builtin(
        [0.5, 2]
    ) == Tensor.from_builtin([2, 1.25])


def test_gradients():
    left = Tensor.from_builtin([2, 3, 4])
    right = Tensor.from_builtin([3, 2, 1])
    result = left / right
    gradients = result.backward(np.array([1, 2, 3]))
    assert np.allclose(gradients[left], [1 / 3, 2 / 2, 3 / 1])
    assert np.allclose(gradients[right], [-2 / 9, -6 / 4, -12 / 1])
