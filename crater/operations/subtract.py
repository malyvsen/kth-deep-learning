from crater.tensor import Tensor
from crater.gradient import Gradients, Gradient
from .coalesce import coalesce


def subtract(left: Tensor, right: Tensor):
    left, right = coalesce(left, right)
    return Tensor.from_numpy(
        data=left.data - right.data,
        backward=lambda gradient: Gradients.accumulate(
            Gradient(tensor=left, gradient=gradient),
            Gradient(tensor=right, gradient=-gradient),
        ),
    )


Tensor.__sub__ = subtract
