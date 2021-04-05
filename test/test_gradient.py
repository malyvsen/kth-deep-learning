import numpy as np
from crater import Tensor, Gradients
from crater import operations as ops


def test_chain():
    tensor = Tensor.from_builtin([1, 2, 3])
    output = -ops.sum(tensor)
    gradients = Gradients.trace(output)

    assert np.allclose(gradients[tensor], [-1, -1, -1])
    assert np.allclose(gradients[output], 1)


def test_accumulate():
    start = Tensor.from_builtin([1, 2, 3])
    intermediate = start * start
    end = ops.sum(start - intermediate)
    gradients = Gradients.trace(end)

    assert np.allclose(gradients[start], [-1, -3, -5])
    assert np.allclose(gradients[intermediate], [-1, -1, -1])
    assert np.allclose(gradients[end], 1)
