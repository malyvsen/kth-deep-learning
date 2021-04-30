import numpy as np
from crater import Tensor, Gradients, Gradient


def test_values():
    assert Tensor.from_builtin([0, 1]).sigmoid() == Tensor.from_builtin(
        [0.5, np.e / (1 + np.e)]
    )


def test_gradients():
    logits = Tensor.from_builtin([1, 2, 3])
    probabilities = logits.sigmoid()
    gradients = Gradients._trace(
        Gradient(tensor=probabilities, gradient=np.array([1, 0, -1]))
    )
    assert np.allclose(gradients[logits], [0.1966, 0, -0.0452], atol=1e-4)
