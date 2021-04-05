import numpy as np
from crater import Tensor, Gradients, Gradient


def test_int():
    assert Tensor.from_builtin(2) + Tensor.from_builtin(3) == Tensor.from_builtin(5)


def test_float():
    assert Tensor.from_builtin(2.5) + Tensor.from_builtin(3) == Tensor.from_builtin(5.5)


def test_list():
    assert Tensor.from_builtin([1, 2.5]) + Tensor.from_builtin(
        [3, -1]
    ) == Tensor.from_builtin([4, 1.5])


def test_broadcast():
    assert Tensor.from_builtin(2) + Tensor.from_builtin([1, 2]) == Tensor.from_builtin(
        [3, 4]
    )


def test_gradients():
    left = Tensor.from_builtin([2, 3, 4])
    right = Tensor.from_builtin([3, 2, 1])
    result = left + right
    gradients = Gradients._trace(Gradient(tensor=result, gradient=np.array([1, 2, 3])))
    assert np.all(gradients[left] == [1, 2, 3])
    assert np.all(gradients[right] == [1, 2, 3])
