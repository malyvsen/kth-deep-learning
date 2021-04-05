import numpy as np
from crater import Tensor, Gradients, Gradient


def test_full():
    assert Tensor.from_builtin([0, 1]).softmax() == Tensor.from_builtin(
        [1 / (1 + np.e), np.e / (1 + np.e)]
    )


def test_one_axis():
    target = [1 / (1 + np.e), np.e / (1 + np.e)]
    assert Tensor.from_builtin([[0, 1], [1, 0]]).softmax(1) == Tensor.from_builtin(
        [target, target[::-1]]
    )


def test_gradients_full():
    logits = Tensor.from_builtin([1, 2, 3])
    probabilities = logits.softmax()
    gradients = Gradients._trace(
        Gradient(tensor=probabilities, gradient=np.array([1, 0, 0]))
    )
    assert np.allclose(gradients[logits], [0.0819, -0.0220, -0.0599], atol=1e-4)


def test_gradients_one_axis():
    logits = Tensor.from_builtin([[1, 2, 3], [4, 5, 6]])
    probabilities = logits.softmax(1)
    gradients = Gradients._trace(
        Gradient(tensor=probabilities, gradient=np.array([[1, 0, 0], [0, 1, 0]]))
    )
    assert np.allclose(
        gradients[logits],
        [[0.0819, -0.0220, -0.0599], [-0.0220, 0.1848, -0.1628]],
        atol=1e-4,
    )
