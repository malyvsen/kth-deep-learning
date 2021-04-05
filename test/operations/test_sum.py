import numpy as np
from crater import Tensor


def test_sum():
    assert Tensor.from_builtin([1, 2, -1]).sum() == Tensor.from_builtin(2)


def test_gradients():
    tensor = Tensor.from_builtin([2, 3, 4])
    result = tensor.sum()
    gradients = result.backward(np.array(-1))
    assert np.all(gradients[tensor] == [-1, -1, -1])
