from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ReLU:
    def forward(self, input: np.ndarray):
        return np.maximum(input, 0)

    def backward(self, gradient: np.ndarray, input: np.ndarray, **kwargs):
        result = gradient.copy()
        result[input < 0] = 0
        return type(self)(), result
