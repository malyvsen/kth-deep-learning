from crater.tensor import Tensor
from crater.gradient import Gradients, Gradient
from .coalesce import coalesce


def negate(tensor: Tensor):
    tensor = coalesce(tensor)
    return Tensor.from_numpy(
        data=-tensor.data,
        backward=lambda gradient: Gradients.accumulate(
            Gradient(tensor=tensor, gradient=-gradient)
        ),
    )


Tensor.__neg__ = negate
