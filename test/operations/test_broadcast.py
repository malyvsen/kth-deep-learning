import numpy as np
import pytest
from crater import Tensor, Gradients
from crater.operations import coalesce


def test_single():
    assert coalesce(Tensor.from_builtin([1, 2, 3])) == Tensor.from_builtin([1, 2, 3])


def test_identical():
    a, b = coalesce(Tensor.from_builtin([1, 2, 3]), Tensor.from_builtin([3, 4, 5]))
    assert a == Tensor.from_builtin([1, 2, 3])
    assert b == Tensor.from_builtin([3, 4, 5])


def test_two():
    a, b = coalesce(
        Tensor.from_builtin([1, 2, 3]), Tensor.from_builtin([[1, 2, 3], [4, 5, 6]])
    )
    assert a == Tensor.from_builtin([[1, 2, 3], [1, 2, 3]])
    assert b == Tensor.from_builtin([[1, 2, 3], [4, 5, 6]])


def test_types():
    a, b, c = coalesce([1, 2, 3], [[1, 2, 3], [4, 5, 6]], 9)
    assert a == Tensor.from_builtin([[1, 2, 3], [1, 2, 3]])
    assert b == Tensor.from_builtin([[1, 2, 3], [4, 5, 6]])
    assert c == Tensor.from_builtin([[9, 9, 9], [9, 9, 9]])


def test_gradients():
    a, b = Tensor.from_builtin([1, 2, 3]), Tensor.from_builtin([[1, 2, 3], [4, 5, 6]])
    x, y = coalesce(a, b)

    x_gradients = Gradients.trace((x * x).sum())
    assert np.all(x_gradients[a] == [4, 8, 12])
    with pytest.raises(KeyError):
        x_gradients[b]

    y_gradients = Gradients.trace((y * y).sum())
    assert np.all(y_gradients[b] == [[2, 4, 6], [8, 10, 12]])
    with pytest.raises(KeyError):
        y_gradients[a]
