from crater.tensor import Tensor
from .coalesce import coalesce


def softmax(tensor: Tensor, axes=None):
    tensor = coalesce(tensor)
    exp_tensor = tensor.exp()
    return exp_tensor / exp_tensor.sum(axes=axes).expand_dims(axes=axes)


Tensor.softmax = softmax
