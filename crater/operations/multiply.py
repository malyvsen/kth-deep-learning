from crater.tensor import Tensor
from crater.gradient import Gradients, Gradient
from .coalesce import coalesce


def multiply(left: Tensor, right: Tensor):
    left, right = coalesce(left, right)
    return Tensor.from_numpy(
        data=left.data * right.data,
        backward=lambda gradient: Gradients.accumulate(
            Gradient(tensor=left, gradient=gradient * right.data),
            Gradient(tensor=right, gradient=gradient * left.data),
        ),
    )


Tensor.__mul__ = multiply
Tensor.__rmul__ = lambda right, left: multiply(left, right)
