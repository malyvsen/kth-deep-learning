import numpy as np
from crater.tensor import Tensor
from .operation import Operation


class Add(Operation):
    def forward(self, left: Tensor, right: Tensor):
        assert left.shape == right.shape
        return Tensor.from_numpy(left.data + right.data)

    def backward(self, gradients: np.ndarray):
        return {
            self.left.id: gradients,
            self.right.id: gradients,
        }


Tensor.__add__ = lambda left, right: Add.apply(left, right)
