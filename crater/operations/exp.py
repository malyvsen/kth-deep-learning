import numpy as np
from crater.tensor import Tensor
from .operation import Operation


class Exp(Operation):
    def forward(self, tensor: Tensor):
        return Tensor.from_numpy(np.exp(tensor.data))

    def backward(self, gradients: np.ndarray):
        return {
            self.tensor.id: gradients * self.output.data,
        }


exp = lambda tensor: Exp.apply(tensor)
