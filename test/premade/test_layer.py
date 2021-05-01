import numpy as np
from crater import Tensor
from crater.premade import Layer


def test_no_activation():
    layer = Layer(
        weights=Tensor.from_builtin([[1, 2, -1], [-3, 5, 0]]),
        biases=Tensor.from_builtin([-1, 1, 2]),
        activation=lambda tensor: tensor,
    )
    assert layer([[0, 1], [3, 1]]) == Tensor.from_builtin([[-4, 6, 2], [-1, 12, -1]])


def test_relu():
    layer = Layer(
        weights=Tensor.from_builtin([[1, 2, -1], [-3, 5, 0]]),
        biases=Tensor.from_builtin([-1, 1, 2]),
        activation=lambda tensor: tensor.clip(low=0),
    )
    assert layer([[0, 1], [3, 1]]) == Tensor.from_builtin([[0, 6, 2], [0, 12, 0]])


def test_from_dims():
    """Does the layer not explode the magnitude of inputs?"""
    layer = Layer.from_dims(in_dim=10, out_dim=20, activation=lambda tensor: tensor)
    output = layer(np.random.normal(size=(100, 10)))
    assert np.isclose(np.mean(output.data), 0, atol=0.25)
    assert np.isclose(np.std(output.data), 1, atol=0.25)
