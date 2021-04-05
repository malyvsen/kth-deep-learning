from crater.tensor import Tensor
from crater.gradient import Gradients, Gradient


def multiply(left: Tensor, right: Tensor):
    assert left.shape == right.shape
    return Tensor.from_numpy(
        data=left.data * right.data,
        backward=lambda gradient: Gradients.accumulate(
            Gradient(tensor=left, gradient=gradient * right.data),
            Gradient(tensor=right, gradient=gradient * left.data),
        ),
    )


Tensor.__mul__ = multiply
