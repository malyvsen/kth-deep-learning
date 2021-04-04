import numpy as np
from crater import Tensor
from crater.operations import Multiply


def test_int():
    assert Tensor.from_builtin(2) * Tensor.from_builtin(3) == Tensor.from_builtin(6)


def test_float():
    assert Tensor.from_builtin(2.5) * Tensor.from_builtin(3) == Tensor.from_builtin(7.5)


def test_list():
    assert Tensor.from_builtin([1, 2.5]) * Tensor.from_builtin(
        [3, -1]
    ) == Tensor.from_builtin([3, -2.5])


def test_backward():
    multiply = Multiply()
    multiply.left = Tensor.from_builtin(2)
    multiply.right = Tensor.from_builtin(3)
    outgoing_gradient = np.array([1, 2, 3])
    computed_gradient = multiply.backward(outgoing_gradient)
    assert computed_gradient.keys() == {multiply.left.id, multiply.right.id}
    assert np.all(computed_gradient[multiply.left.id] == outgoing_gradient * 3)
    assert np.all(computed_gradient[multiply.right.id] == outgoing_gradient * 2)
