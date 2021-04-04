import numpy as np
from crater import Tensor
from crater.operations import Subtract


def test_int():
    assert Tensor.from_builtin(2) - Tensor.from_builtin(3) == Tensor.from_builtin(-1)


def test_float():
    assert Tensor.from_builtin(2.5) - Tensor.from_builtin(3) == Tensor.from_builtin(
        -0.5
    )


def test_list():
    assert Tensor.from_builtin([1, 2.5]) - Tensor.from_builtin(
        [3, -1]
    ) == Tensor.from_builtin([-2, 3.5])


def test_gradients():
    subtract = Subtract()
    subtract.left = Tensor.from_builtin([2, 3, 4])
    subtract.right = Tensor.from_builtin([3, 2, 1])
    subtract.output = Tensor.from_builtin([-1, 1, 3])
    outgoing_gradient = np.array([1, 2, 3])
    computed_gradient = subtract.gradients(outgoing_gradient)
    assert computed_gradient.keys() == {subtract.left.id, subtract.right.id}
    assert np.all(computed_gradient[subtract.left.id] == outgoing_gradient)
    assert np.all(computed_gradient[subtract.right.id] == -outgoing_gradient)
