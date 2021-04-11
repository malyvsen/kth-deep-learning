from typing import Callable
from dataclasses import dataclass
import numpy as np
from .tensor import Tensor
from .operations import matrix_multiply


@dataclass(frozen=True)
class Layer:
    weights: Tensor
    biases: Tensor
    activation: Callable[[Tensor], Tensor]

    @classmethod
    def from_dims(cls, in_dim, out_dim, activation):
        return cls(
            weights=Tensor.from_numpy(
                np.random.normal(scale=in_dim ** -0.5, size=(in_dim, out_dim))
            ),
            biases=Tensor.from_numpy(np.zeros(out_dim)),
            activation=activation,
        )

    def __call__(self, input: Tensor):
        return self.activation(matrix_multiply(input, self.weights) + self.biases)
