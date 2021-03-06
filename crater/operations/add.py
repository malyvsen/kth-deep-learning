from crater.tensor import Tensor
from crater.gradient import Gradients, Gradient
from .coalesce import coalesce


def add(left: Tensor, right: Tensor):
    left, right = coalesce(left, right)
    return Tensor.from_numpy(
        data=left.data + right.data,
        backward=lambda gradient: Gradients.accumulate(
            Gradient(tensor=left, gradient=gradient),
            Gradient(tensor=right, gradient=gradient),
        ),
    )


Tensor.__add__ = add
Tensor.__radd__ = lambda right, left: add(left, right)
