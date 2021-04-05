from crater.tensor import Tensor
from crater.gradient import Gradients, Gradient


def add(left: Tensor, right: Tensor):
    assert left.shape == right.shape
    return Tensor.from_numpy(
        data=left.data + right.data,
        backward=lambda gradient: Gradients.accumulate(
            Gradient(tensor=left, gradient=gradient),
            Gradient(tensor=right, gradient=gradient),
        ),
    )


Tensor.__add__ = add
