import numpy as np
from crater import Tensor


def test_e():
    assert Tensor.from_builtin(1).exp() == Tensor.from_builtin(np.e)


def test_gradients():
    tensor = Tensor.from_builtin([2, 3, 4])
    result = tensor.exp()
    gradients = result.backward(np.array([1, 2, 3]))
    assert np.all(gradients[tensor] == np.exp([2, 3, 4]) * [1, 2, 3])
