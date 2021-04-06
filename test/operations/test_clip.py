import pytest
import numpy as np
from crater import Tensor, Gradients, Gradient


def test_builtin():
    assert Tensor.from_builtin(2).clip(0, 5) == Tensor.from_builtin(2)
    assert Tensor.from_builtin(-1).clip(0, 5) == Tensor.from_builtin(0)
    assert Tensor.from_builtin(6).clip(0, 5) == Tensor.from_builtin(5)


def test_tensor():
    low = Tensor.from_builtin(0)
    high = Tensor.from_builtin(5)
    assert Tensor.from_builtin(2).clip(low, high) == Tensor.from_builtin(2)
    assert Tensor.from_builtin(-1).clip(low, high) == low
    assert Tensor.from_builtin(6).clip(low, high) == high


def test_mixed():
    assert Tensor.from_builtin(2).clip(0, [1, 2, 3]) == Tensor.from_builtin([1, 2, 2])
    assert Tensor.from_builtin(-1).clip(
        Tensor.from_builtin(0), 5
    ) == Tensor.from_builtin(0)


def test_partial():
    assert Tensor.from_builtin(2).clip() == Tensor.from_builtin(2)
    assert Tensor.from_builtin(-1).clip(low=0) == Tensor.from_builtin(0)
    assert Tensor.from_builtin(6).clip(high=5) == Tensor.from_builtin(5)


def test_gradients():
    tensor = Tensor.from_builtin([2, 3, 4])
    low = Tensor.from_builtin([1, 2, 1])
    high = Tensor.from_builtin([3, 4, 2])
    result = tensor.clip(low, high)
    gradients = Gradients._trace(Gradient(tensor=result, gradient=np.array([1, 2, 3])))
    assert np.allclose(gradients[tensor], [1, 2, 0])
    with pytest.raises(KeyError):
        gradients[low]
    with pytest.raises(KeyError):
        gradients[high]
