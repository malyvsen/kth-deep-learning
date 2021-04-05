import numpy as np
from crater import Tensor
from crater.operations import sum


def test_sum():
    assert sum(Tensor.from_builtin([1, 2, -1])) == Tensor.from_builtin(2)


def test_gradients():
    tensor = Tensor.from_builtin([2, 3, 4])
    result = sum(tensor)
    gradients = result.backward(np.array(-1))
    assert np.all(gradients[tensor] == [-1, -1, -1])
