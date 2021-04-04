import numpy as np
from crater import Tensor
from crater.operations import Add


def test_int():
    assert Tensor.from_builtin(2) + Tensor.from_builtin(3) == Tensor.from_builtin(5)


def test_float():
    assert Tensor.from_builtin(2.5) + Tensor.from_builtin(3) == Tensor.from_builtin(5.5)


def test_list():
    assert Tensor.from_builtin([1, 2.5]) + Tensor.from_builtin(
        [3, -1]
    ) == Tensor.from_builtin([4, 1.5])


def test_gradients():
    add = Add()
    add.left = Tensor.from_builtin([2, 3, 4])
    add.right = Tensor.from_builtin([3, 2, 1])
    add.output = Tensor.from_builtin([5, 5, 5])
    outgoing_gradient = np.array([1, 2, 3])
    computed_gradient = add.gradients(outgoing_gradient)
    assert computed_gradient.keys() == {add.left.id, add.right.id}
    assert np.all(computed_gradient[add.left.id] == outgoing_gradient)
    assert np.all(computed_gradient[add.right.id] == outgoing_gradient)
