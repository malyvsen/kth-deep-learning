from typing import Dict
from dataclasses import dataclass
import numpy as np
from crater.tensor import Tensor
from .gradient import Gradient


@dataclass(frozen=True)
class Gradients:
    gradients: Dict[int, Gradient]

    @classmethod
    def accumulate(cls, *gradients: Gradient) -> "Gradients":
        result = {}
        for gradient in gradients:
            tensor_id = id(gradient.tensor)
            result[tensor_id] = Gradient(
                tensor=gradient.tensor,
                gradient=(
                    result[tensor_id].gradient + gradient.gradient
                    if tensor_id in result
                    else gradient.gradient
                ),
            )
        return cls(gradients=result)

    @classmethod
    def trace(cls, tensor: Tensor) -> "Gradients":
        if not all(dim == 1 for dim in tensor.shape):
            raise ValueError("Can only compute gradients for single-item tensor")
        return cls._trace(
            gradient=Gradient(tensor=tensor, gradient=np.ones_like(tensor.data))
        )

    @classmethod
    def _trace(cls, gradient) -> "Gradients":
        result = cls.accumulate(gradient)
        if gradient.tensor.backward is None:
            return result

        for connected_gradient in gradient.tensor.backward(gradient.gradient):
            result = cls.accumulate(*result, *cls._trace(gradient=connected_gradient))

        return result

    def __iter__(self):
        return iter(self.gradients.values())

    def __getitem__(self, key: Tensor):
        return self.gradients[id(key)].gradient
