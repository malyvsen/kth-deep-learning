import numpy as np
from crater.tensor import Tensor
from crater.gradient import Gradients, Gradient


def matrix_multiply(left: Tensor, right: Tensor):
    assert len(left.shape) == 2
    assert len(right.shape) == 2
    assert left.shape[1] == right.shape[0]
    return Tensor.from_numpy(
        data=np.matmul(left.data, right.data),
        backward=lambda gradient: Gradients.accumulate(
            Gradient(tensor=left, gradient=np.matmul(gradient, right.data.T)),
            Gradient(tensor=right, gradient=np.matmul(left.data.T, gradient)),
        ),
    )
