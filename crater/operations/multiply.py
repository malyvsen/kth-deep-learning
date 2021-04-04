import numpy as np
from crater.tensor import Tensor
from .operation import Operation


class Multiply(Operation):
    def forward(self, left: Tensor, right: Tensor):
        assert left.shape == right.shape
        return Tensor.from_numpy(left.data * right.data)

    def backward(self, gradients: np.ndarray):
        return {
            self.left.id: gradients * self.right.data,
            self.right.id: gradients * self.left.data,
        }


Tensor.__mul__ = lambda left, right: Multiply.apply(left, right)
