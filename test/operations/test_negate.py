import numpy as np
from crater import Tensor
from crater.operations import Negate


def test_int():
    assert -Tensor.from_builtin(3) == Tensor.from_builtin(-3)


def test_float():
    assert -Tensor.from_builtin(2.5) == Tensor.from_builtin(-2.5)


def test_list():
    assert -Tensor.from_builtin([-1, 2.5]) == Tensor.from_builtin([1, -2.5])


def test_backward():
    negate = Negate()
    negate.tensor = Tensor.from_builtin(2)
    outgoing_gradient = np.array([1, 2, 3])
    computed_gradient = negate.backward(outgoing_gradient)
    assert computed_gradient.keys() == {negate.tensor.id}
    assert np.all(computed_gradient[negate.tensor.id] == -outgoing_gradient)
