from typing import Tuple
import numpy as np
from crater.tensor import Tensor
from crater.gradient import Gradients, Gradient


def expand_dims(tensor: Tensor, axes: Tuple[int]):
    return Tensor.from_numpy(
        data=np.expand_dims(tensor.data, axes),
        backward=lambda gradient: Gradients.accumulate(
            Gradient(tensor=tensor, gradient=np.squeeze(gradient, axes))
        ),
    )


Tensor.expand_dims = expand_dims
