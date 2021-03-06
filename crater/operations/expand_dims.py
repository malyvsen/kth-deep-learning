from typing import Union, Tuple
import numpy as np
from crater.tensor import Tensor
from crater.gradient import Gradients, Gradient
from crater.utils import tuplify
from .coalesce import coalesce


def expand_dims(tensor: Tensor, axes: Union[None, int, Tuple[int]] = None):
    tensor = coalesce(tensor)
    axes = () if axes is None else tuplify(axes)
    return Tensor.from_numpy(
        data=np.expand_dims(tensor.data, axes),
        backward=lambda gradient: Gradients.accumulate(
            Gradient(tensor=tensor, gradient=np.squeeze(gradient, axes))
        ),
    )


Tensor.expand_dims = expand_dims
