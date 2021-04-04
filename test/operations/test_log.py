import numpy as np
from crater import Tensor
from crater.operations import Log, log


def test_e():
    assert log(Tensor.from_builtin(np.e)) == Tensor.from_builtin(1)


def test_gradients():
    log = Log()
    log.tensor = Tensor.from_builtin([2, 3, 4])
    log.output = Tensor.from_numpy(np.log([2, 3, 4]))
    outgoing_gradient = np.array([1, 2, 3])
    computed_gradient = log.gradients(outgoing_gradient)
    assert computed_gradient.keys() == {log.tensor.id}
    assert np.all(computed_gradient[log.tensor.id] == outgoing_gradient / [2, 3, 4])
