import numpy as np
from crater import Tensor
from crater.operations import MatrixMultiply, matrix_multiply


def test_matrix_multiply():
    assert (
        matrix_multiply(
            Tensor.from_builtin([[1, 2, 3], [3, 4, 1]]),
            Tensor.from_builtin([[3], [1], [-1]]),
        )
        == Tensor.from_builtin([[2], [12]])
    )


def test_gradients():
    matrix_multiply = MatrixMultiply()
    matrix_multiply.left = Tensor.from_builtin([[1, 2, 3], [3, 4, 1]])
    matrix_multiply.right = Tensor.from_builtin([[3], [1], [-1]])
    matrix_multiply.output = Tensor.from_builtin([[2], [12]])
    outgoing_gradient = np.array([[1], [-1]])
    computed_gradient = matrix_multiply.gradients(outgoing_gradient)
    assert computed_gradient.keys() == {
        matrix_multiply.left.id,
        matrix_multiply.right.id,
    }
    assert np.all(
        computed_gradient[matrix_multiply.left.id] == [[3, 1, -1], [-3, -1, 1]]
    )
    assert np.all(computed_gradient[matrix_multiply.right.id] == [[-2], [-2], [2]])
