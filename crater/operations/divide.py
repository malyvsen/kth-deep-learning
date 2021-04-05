import numpy as np
from crater.tensor import Tensor
from crater.gradient import Gradients, Gradient
from .broadcast import broadcast


def divide(left: Tensor, right: Tensor):
    left, right = broadcast(left, right)
    return Tensor.from_numpy(
        data=left.data / right.data,
        backward=lambda gradient: Gradients.accumulate(
            Gradient(tensor=left, gradient=gradient / right.data),
            Gradient(
                tensor=right,
                gradient=-gradient * left.data / np.square(right.data),
            ),
        ),
    )


Tensor.__truediv__ = divide
