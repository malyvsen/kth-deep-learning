import numpy as np
from crater.tensor import Tensor
from .operation import Operation


class MatrixMultiply(Operation):
    def forward(self, left: Tensor, right: Tensor):
        assert len(left.shape) == 2
        assert len(right.shape) == 2
        assert left.shape[1] == right.shape[0]
        return Tensor.from_numpy(np.matmul(left.data, right.data))

    def backward(self, gradients: np.ndarray):
        return {
            self.left.id: np.matmul(gradients, self.right.data.T),
            self.right.id: np.matmul(self.left.data.T, gradients),
        }


matrix_multiply = lambda left, right: MatrixMultiply.apply(left, right)
