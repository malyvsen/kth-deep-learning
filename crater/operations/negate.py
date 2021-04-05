from crater.tensor import Tensor
from crater.gradient import Gradients, Gradient


def negate(tensor: Tensor):
    return Tensor.from_numpy(
        data=-tensor.data,
        backward=lambda gradient: Gradients.accumulate(
            Gradient(tensor=tensor, gradient=-gradient)
        ),
    )


Tensor.__neg__ = negate
