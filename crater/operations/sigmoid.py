from crater.tensor import Tensor
from .coalesce import coalesce


def sigmoid(tensor: Tensor):
    tensor = coalesce(tensor)
    exp_tensor = tensor.exp()
    return exp_tensor / (exp_tensor + 1)


Tensor.sigmoid = sigmoid
