import numpy as np
from crater import Tensor
from crater.operations import Sum, sum


def test_sum():
    assert sum(Tensor.from_builtin([1, 2, -1])) == Tensor.from_builtin(2)


def test_gradients():
    sum = Sum()
    sum.tensor = Tensor.from_builtin([2, 3, 4])
    sum.output = Tensor.from_builtin(9)
    outgoing_gradient = np.array(-1)
    computed_gradient = sum.gradients(outgoing_gradient)
    assert computed_gradient.keys() == {sum.tensor.id}
    assert np.all(computed_gradient[sum.tensor.id] == [-1, -1, -1])
