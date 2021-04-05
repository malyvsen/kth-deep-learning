import numpy as np
from crater import Tensor
from crater.operations import log


def test_e():
    assert log(Tensor.from_builtin(np.e)) == Tensor.from_builtin(1)


def test_gradients():
    tensor = Tensor.from_builtin([2, 3, 4])
    result = log(tensor)
    gradients = result.backward(np.array([1, 2, 3]))
    assert np.all(gradients[tensor] == np.array([1 / 2, 2 / 3, 3 / 4]))
