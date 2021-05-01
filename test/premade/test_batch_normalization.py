import pytest
import numpy as np
from crater import Tensor, Gradients
from crater.premade import BatchNormalization


def test_no_mode():
    batch_norm = BatchNormalization.from_shape(2)
    with pytest.raises(ValueError):
        batch_norm(Tensor.from_builtin([[1, 2], [3, 4]]))


def test_mode_switch():
    batch_norm = BatchNormalization.from_shape(1, persistence=0)
    with BatchNormalization.mode(BatchNormalization.Mode.train):
        assert batch_norm([[0], [4]]) == Tensor.from_builtin([[-1], [1]])
    with BatchNormalization.mode(BatchNormalization.Mode.test):
        assert batch_norm([[1], [5]]) == Tensor.from_builtin([[-0.5], [1.5]])


def test_gradients():
    tensor = Tensor.from_builtin([[1, 2], [3, 4]])
    batch_norm = BatchNormalization(
        mean=np.array([4, -1]),
        standard_deviation=np.array([1, 0.5]),
        persistence=0.9,
        shift=Tensor.from_builtin([3, 2]),
        scale=Tensor.from_builtin([1, 1]),
    )
    with BatchNormalization.mode(BatchNormalization.Mode.test):
        loss = batch_norm(tensor).sum()
    gradients = Gradients.trace(loss)
    assert np.allclose(gradients[tensor], [[1, 2], [1, 2]])
    assert np.allclose(gradients[batch_norm.shift], [2, 2])
    assert np.allclose(gradients[batch_norm.scale], [-4, 16])
