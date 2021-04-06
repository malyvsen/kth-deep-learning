import numpy as np
from crater.tensor import Tensor
from crater.gradient import Gradients, Gradient
from .coalesce import coalesce


def exp(tensor: Tensor):
    tensor = coalesce(tensor)
    return Tensor.from_numpy(
        data=np.exp(tensor.data),
        backward=lambda gradient: Gradients.accumulate(
            Gradient(tensor=tensor, gradient=gradient * np.exp(tensor.data))
        ),
    )


Tensor.exp = exp
