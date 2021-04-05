import numpy as np
from crater.tensor import Tensor
from crater.gradient import Gradients, Gradient


def sum(tensor: Tensor):
    return Tensor.from_numpy(
        data=tensor.data.sum(),
        backward=lambda gradient: Gradients.accumulate(
            Gradient(tensor=tensor, gradient=np.full_like(tensor.data, gradient))
        ),
    )


Tensor.sum = sum
