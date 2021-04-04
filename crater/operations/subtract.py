import numpy as np
from crater.tensor import Tensor
from .operation import Operation


class Subtract(Operation):
    def forward(self, left: Tensor, right: Tensor):
        assert left.shape == right.shape
        return Tensor.from_numpy(left.data - right.data)

    def backward(self, gradients: np.ndarray):
        return {
            self.left.id: gradients,
            self.right.id: -gradients,
        }


Tensor.__sub__ = lambda left, right: Subtract.apply(left, right)
