from crater.tensor import Tensor
from crater.gradient import Gradients, Gradient
from .broadcast import broadcast


def subtract(left: Tensor, right: Tensor):
    left, right = broadcast(left, right)
    return Tensor.from_numpy(
        data=left.data - right.data,
        backward=lambda gradient: Gradients.accumulate(
            Gradient(tensor=left, gradient=gradient),
            Gradient(tensor=right, gradient=-gradient),
        ),
    )


Tensor.__sub__ = subtract
