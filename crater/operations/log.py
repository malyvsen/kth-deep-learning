import numpy as np
from crater.tensor import Tensor
from crater.gradient import Gradients, Gradient
from .coalesce import coalesce


def log(tensor: Tensor):
    tensor = coalesce(tensor)
    return Tensor.from_numpy(
        data=np.log(tensor.data),
        backward=lambda gradient: Gradients.accumulate(
            Gradient(tensor=tensor, gradient=gradient / tensor.data)
        ),
    )


Tensor.log = log
