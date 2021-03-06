import numpy as np
from crater.tensor import Tensor
from crater.gradient import Gradients, Gradient
from crater.utils import tuplify
from .coalesce import coalesce


def sum(tensor: Tensor, axes=None):
    tensor = coalesce(tensor)
    axes = tuplify(range(len(tensor.shape)) if axes is None else axes)
    return Tensor.from_numpy(
        data=tensor.data.sum(axes),
        backward=lambda gradient: Gradients.accumulate(
            Gradient(
                tensor=tensor,
                gradient=np.tile(
                    np.expand_dims(gradient, axes),
                    [
                        dim if idx in axes or idx - len(tensor.shape) in axes else 1
                        for idx, dim in enumerate(tensor.shape)
                    ],
                ),
            )
        ),
    )


Tensor.sum = sum
