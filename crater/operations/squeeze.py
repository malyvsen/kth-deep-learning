from typing import Union, Tuple
import numpy as np
from crater.tensor import Tensor
from crater.gradient import Gradients, Gradient
from crater.utils import tuplify


def squeeze(tensor: Tensor, axes: Union[None, int, Tuple[int]]):
    axes = () if axes is None else tuplify(axes)
    return Tensor.from_numpy(
        data=np.squeeze(tensor.data, axes),
        backward=lambda gradient: Gradients.accumulate(
            Gradient(tensor=tensor, gradient=np.expand_dims(gradient, axes))
        ),
    )


Tensor.squeeze = squeeze
