from dataclasses import dataclass
import numpy as np


class Layer:
    @classmethod
    def _xavier_weights(cls, in_dim, out_dim):
        avg_dim = (in_dim + out_dim) / 2
        return np.random.normal(scale=avg_dim ** -0.5, size=(in_dim, out_dim))


@dataclass(frozen=True)
class PureLayer(Layer):
    weights: np.ndarray

    @classmethod
    def xavier(cls, in_dim, out_dim):
        return cls(weights=cls._xavier_weights(in_dim, out_dim))

    def forward(self, input: np.ndarray) -> np.ndarray:
        return np.matmul(input, self.weights)

    def backward(self, gradient: np.ndarray, input: np.ndarray, **kwargs):
        input_gradient, weights_gradient = matmul_backward(
            gradient, left=input, right=self.weights
        )
        return (
            type(self)(
                weights=weights_gradient - 2 * kwargs["regularization"] * self.weights
            ),
            input_gradient,
        )


@dataclass(frozen=True)
class BiasedLayer(Layer):
    weights: np.ndarray
    biases: np.ndarray

    @classmethod
    def xavier(cls, in_dim, out_dim):
        return cls(
            weights=cls._xavier_weights(in_dim, out_dim), biases=np.zeros(out_dim)
        )

    def forward(self, input: np.ndarray) -> np.ndarray:
        return np.matmul(input, self.weights) + self.biases

    def backward(self, gradient: np.ndarray, input: np.ndarray, **kwargs):
        input_gradient, weights_gradient = matmul_backward(
            gradient, left=input, right=self.weights
        )
        return (
            type(self)(
                weights=weights_gradient - 2 * kwargs["regularization"] * self.weights,
                biases=gradient.sum(axis=0),
            ),
            input_gradient,
        )


def matmul_backward(gradient: np.ndarray, left: np.ndarray, right: np.ndarray):
    return (
        np.matmul(gradient, right.T),  # gradient wrt left operand
        np.matmul(left.T, gradient),  # gradient wrt right operand
    )
