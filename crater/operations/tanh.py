from crater.tensor import Tensor
from .coalesce import coalesce


def tanh(tensor: Tensor):
    tensor = coalesce(tensor)
    exp_tensor = (2 * tensor).exp()
    return (exp_tensor - 1) / (exp_tensor + 1)


Tensor.tanh = tanh
