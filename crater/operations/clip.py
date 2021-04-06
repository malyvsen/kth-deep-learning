import numpy as np
from crater.tensor import Tensor
from crater.gradient import Gradients, Gradient
from .coalesce import coalesce


def clip(tensor: Tensor, low=None, high=None):
    if low is None:
        low = -float("inf")
    if high is None:
        high = float("inf")
    tensor, low, high = coalesce(tensor, low, high)

    def backward(gradient: np.ndarray):
        result = gradient.copy()
        result[(tensor.data < low.data) | (tensor.data > high.data)] = 0
        return Gradients.accumulate(Gradient(tensor=tensor, gradient=result))

    return Tensor.from_numpy(
        data=np.clip(tensor.data, low.data, high.data),
        backward=backward,
    )


Tensor.clip = clip
