from typing import Callable
from dataclasses import dataclass
import numpy as np
from crater.tensor import Tensor
from crater.operations import matrix_multiply
from .batch_normalization import BatchNormalization


@dataclass(frozen=True)
class Layer:
    weights: Tensor
    biases: Tensor
    batch_normalization: BatchNormalization
    activation: Callable[[Tensor], Tensor]

    @classmethod
    def from_dims(
        cls, in_dim, out_dim, activation, persistence: float = None
    ) -> "Layer":
        return cls(
            weights=Tensor.from_numpy(
                np.random.normal(scale=in_dim ** -0.5, size=(in_dim, out_dim))
            ),
            biases=Tensor.from_numpy(np.zeros(out_dim)),
            batch_normalization=BatchNormalization.from_shape(
                shape=(out_dim,), persistence=persistence
            ),
            activation=activation,
        )

    def __call__(self, input: Tensor) -> Tensor:
        return self.activation(matrix_multiply(input, self.weights) + self.biases)
