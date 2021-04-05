import numpy as np
from crater import Tensor, Gradients, Gradient
from crater.operations import softmax


def test_zero_one():
    assert softmax(Tensor.from_builtin([0, 1])) == Tensor.from_builtin(
        [1 / (1 + np.e), np.e / (1 + np.e)]
    )


def test_gradients():
    logits = Tensor.from_builtin([1, 2, 3])
    probabilities = softmax(logits)
    gradients = Gradients._trace(
        Gradient(tensor=probabilities, gradient=np.array([1, 0, 0]))
    )
    assert np.allclose(gradients[logits], [0.0819, -0.0220, -0.0599], atol=1e-4)
