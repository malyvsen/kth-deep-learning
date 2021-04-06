import numpy as np
from crater import Tensor
from crater.operations import matrix_multiply


def test_tensor():
    assert (
        matrix_multiply(
            Tensor.from_builtin([[1, 2, 3], [3, 4, 1]]),
            Tensor.from_builtin([[3], [1], [-1]]),
        )
        == Tensor.from_builtin([[2], [12]])
    )


def test_list():
    assert (
        matrix_multiply(
            [[1, 2, 3], [3, 4, 1]],
            [[3], [1], [-1]],
        )
        == Tensor.from_builtin([[2], [12]])
    )


def test_gradients():
    left = Tensor.from_builtin([[1, 2, 3], [3, 4, 1]])
    right = Tensor.from_builtin([[3], [1], [-1]])
    output = matrix_multiply(left, right)
    gradients = output.backward(np.array([[1], [-1]]))
    assert np.all(gradients[left] == [[3, 1, -1], [-3, -1, 1]])
    assert np.all(gradients[right] == [[-2], [-2], [2]])
