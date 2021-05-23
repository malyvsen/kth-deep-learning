from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Softmax:
    def forward(self, input: np.ndarray):
        exp_input = np.exp(input)
        return exp_input / exp_input.sum(axis=1, keepdims=True)

    def backward(self, gradient: np.ndarray, input: np.ndarray, **kwargs):
        output = self.forward(input)
        return type(self)(), output * (
            gradient - np.sum(gradient * output, axis=1, keepdims=True)
        )
