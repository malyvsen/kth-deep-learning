import numpy as np
from crater import Tensor


def test_none():
    assert Tensor.from_builtin([-1, 2]).expand_dims() == Tensor.from_builtin([-1, 2])


def test_single():
    assert Tensor.from_builtin([-1, 2]).expand_dims(0) == Tensor.from_builtin([[-1, 2]])
    assert Tensor.from_builtin([-1, 2]).expand_dims(1) == Tensor.from_builtin(
        [[-1], [2]]
    )


def test_list():
    assert Tensor.from_builtin([-1, 2]).expand_dims([0, 2]) == Tensor.from_builtin(
        [[[-1], [2]]]
    )


def test_gradients():
    tensor = Tensor.from_builtin([2, 3, 4])
    result = tensor.expand_dims(1)
    gradients = result.backward(np.array([[1], [2], [3]]))
    assert np.all(gradients[tensor] == [1, 2, 3])
