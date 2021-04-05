import numpy as np
from crater import Tensor


def test_e():
    assert Tensor.from_builtin(np.e).log() == Tensor.from_builtin(1)


def test_gradients():
    tensor = Tensor.from_builtin([2, 3, 4])
    result = tensor.log()
    gradients = result.backward(np.array([1, 2, 3]))
    assert np.all(gradients[tensor] == np.array([1 / 2, 2 / 3, 3 / 4]))
