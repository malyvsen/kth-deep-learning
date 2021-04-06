from typing import Tuple
import numpy as np
from crater.tensor import Tensor
from crater.gradient import Gradients, Gradient
from .coalesce import coalesce


def tile(tensor: Tensor, tiling: Tuple[int]):
    tensor = coalesce(tensor)
    tiling = tuple(tiling)
    assert len(tensor.shape) == len(tiling)
    return Tensor.from_numpy(
        data=np.tile(tensor.data, tiling),
        backward=lambda gradient: Gradients.accumulate(
            Gradient(
                tensor=tensor,
                gradient=np.reshape(
                    gradient,
                    [
                        value
                        for idx, dim in enumerate(tensor.shape)
                        for value in [tiling[idx], dim]
                    ],
                ).sum(tuple(range(0, len(tiling) * 2, 2))),
            )
        ),
    )


Tensor.tile = tile
