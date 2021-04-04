import numpy as np
from crater import Tensor
from crater.operations import Exp, exp


def test_e():
    assert exp(Tensor.from_builtin(1)) == Tensor.from_builtin(np.e)


def test_gradients():
    exp = Exp()
    exp.tensor = Tensor.from_builtin([2, 3, 4])
    exp.output = Tensor.from_numpy(np.exp([2, 3, 4]))
    outgoing_gradient = np.array([1, 2, 3])
    computed_gradient = exp.gradients(outgoing_gradient)
    assert computed_gradient.keys() == {exp.tensor.id}
    assert np.all(
        computed_gradient[exp.tensor.id] == outgoing_gradient * np.exp([2, 3, 4])
    )
