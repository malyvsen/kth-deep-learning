import numpy as np
from crater import Tensor


def test_list():
    assert Tensor.from_builtin([-1, 2]).tile([3]) == Tensor.from_builtin(
        [-1, 2, -1, 2, -1, 2]
    )


def test_matrix():
    assert Tensor.from_builtin([[1, 2], [3, 4]]).tile([2, 1]) == Tensor.from_builtin(
        [[1, 2], [3, 4], [1, 2], [3, 4]]
    )
    assert Tensor.from_builtin([[1, 2], [3, 4]]).tile([1, 2]) == Tensor.from_builtin(
        [[1, 2, 1, 2], [3, 4, 3, 4]]
    )


def test_list_gradients():
    tensor = Tensor.from_builtin([2, 3, 4])
    result = tensor.tile([2])
    gradients = result.backward(np.array([1, 2, 3, 4, 5, 6]))
    assert np.all(gradients[tensor] == [5, 7, 9])


def test_matrix_gradients():
    tensor = Tensor.from_builtin([[5, 6], [7, 8]])
    result = tensor.tile([2, 3])
    gradients = result.backward(
        np.array(
            [
                [1, 2, 5, 3, 1, 0],
                [3, 5, 9, 2, 1, 4],
                [6, 2, 5, 3, 3, 2],
                [2, 5, 7, 2, 4, 7],
            ]
        )
    )
    assert np.all(gradients[tensor] == [[21, 12], [26, 25]])
