import numpy as np
from crater.tensor import Tensor
from .operation import Operation


class Log(Operation):
    def forward(self, tensor: Tensor):
        return Tensor.from_numpy(np.log(tensor.data))

    def backward(self, gradients: np.ndarray):
        return {
            self.tensor.id: gradients / self.tensor.data,
        }


log = lambda tensor: Log.apply(tensor)
