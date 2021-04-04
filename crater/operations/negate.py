import numpy as np
from crater.tensor import Tensor
from .operation import Operation


class Negate(Operation):
    def forward(self, tensor: Tensor):
        return Tensor.from_numpy(-tensor.data)

    def backward(self, gradients: np.ndarray):
        return {
            self.tensor.id: -gradients,
        }


Tensor.__neg__ = lambda tensor: Negate.apply(tensor)
