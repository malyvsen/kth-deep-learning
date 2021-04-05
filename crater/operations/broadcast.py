from typing import Tuple
import numpy as np
from crater.tensor import Tensor
from crater.gradient import Gradients, Gradient


def broadcast(*tensors: Tensor):
    if len(tensors) == 1:
        return tensors[0]
    target_shape = np.broadcast(*[tensor.data for tensor in tensors]).shape
    expanded = [
        tensor.expand_dims(list(range(len(target_shape) - len(tensor.shape))))
        for tensor in tensors
    ]
    return [
        tensor.tile(
            [
                target_dim // tensor_dim
                for target_dim, tensor_dim in zip(target_shape, tensor.shape)
            ]
        )
        for tensor in expanded
    ]
