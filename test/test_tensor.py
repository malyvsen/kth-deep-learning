import pytest
from dataclasses import FrozenInstanceError
from crater import Tensor
import numpy as np


def test_equal_int():
    assert Tensor.from_builtin(2) == Tensor.from_builtin(2)


def test_equal_list():
    assert Tensor.from_builtin([1, 2, 4]) == Tensor.from_builtin([1, 2, 4])


def test_equal_array():
    assert Tensor.from_numpy(np.array([1, 2, 4.5])) == Tensor.from_numpy(
        np.array([1, 2, 4.5])
    )


def test_unequal_int():
    assert Tensor.from_builtin(2) != Tensor.from_builtin(3)


def test_unequal_list():
    assert Tensor.from_builtin([1, 2, 3]) != Tensor.from_builtin([1, 2, 4])


def test_unequal_array():
    assert Tensor.from_numpy(np.array([1, 2, 4.5])) != Tensor.from_numpy(
        np.array([1, 2, 5.5])
    )


def test_shape():
    assert Tensor.from_builtin(15).shape == ()
    assert Tensor.from_builtin([15]).shape == (1,)
    assert Tensor.from_builtin([15, 10]).shape == (2,)
    assert Tensor.from_builtin([[2, 3, 4], [0, 9, 1]]).shape == (2, 3)


def test_read_only():
    with pytest.raises(FrozenInstanceError):
        tensor = Tensor.from_builtin([1, 2, 3])
        tensor.data = 3

    with pytest.raises(ValueError):
        tensor = Tensor.from_builtin([1, 2, 3])
        tensor.data[1] = 3


def test_convert():
    tensor = Tensor.convert([1, 2])
    assert tensor == Tensor.from_numpy(np.array([1, 2]))
    assert tensor == Tensor.from_builtin([1, 2])
    assert tensor is Tensor.convert(tensor)


def test_numpy_copy():
    tensor = Tensor.from_builtin([1, -1])
    array = tensor.numpy
    array[0] = 3
    assert np.all(array == [3, -1])
    assert tensor == Tensor.from_builtin([1, -1])
