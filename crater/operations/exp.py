import numpy as np
from crater.tensor import Tensor
from crater.gradient import Gradients, Gradient


def exp(tensor: Tensor):
    return Tensor.from_numpy(
        data=np.exp(tensor.data),
        backward=lambda gradient: Gradients.accumulate(
            Gradient(tensor=tensor, gradient=gradient * np.exp(tensor.data))
        ),
    )
