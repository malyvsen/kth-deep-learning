import numpy as np
from crater import Tensor
from crater.operations import exp


def test_e():
    assert exp(Tensor.from_builtin(1)) == Tensor.from_builtin(np.e)


def test_gradients():
    tensor = Tensor.from_builtin([2, 3, 4])
    result = exp(tensor)
    gradients = result.backward(np.array([1, 2, 3]))
    assert np.all(gradients[tensor] == np.exp([2, 3, 4]) * [1, 2, 3])
