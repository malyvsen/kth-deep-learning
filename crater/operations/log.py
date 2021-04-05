import numpy as np
from crater.tensor import Tensor
from crater.gradient import Gradients, Gradient


def log(tensor: Tensor):
    return Tensor.from_numpy(
        data=np.log(tensor.data),
        backward=lambda gradient: Gradients.accumulate(
            Gradient(tensor=tensor, gradient=gradient / tensor.data)
        ),
    )
