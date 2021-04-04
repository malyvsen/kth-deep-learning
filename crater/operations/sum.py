import numpy as np
from crater.tensor import Tensor
from .operation import Operation


class Sum(Operation):
    def forward(self, tensor: Tensor):
        return Tensor.from_numpy(np.sum(tensor.data))

    def backward(self, gradients: np.ndarray):
        return {
            self.tensor.id: gradients * np.ones_like(self.tensor.data),
        }


sum = lambda tensor: Sum.apply(tensor)
