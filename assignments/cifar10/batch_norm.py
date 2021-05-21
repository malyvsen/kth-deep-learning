from typing import Annotated
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=False)
class BatchNorm:
    mean: Annotated[np.ndarray, "Estimate of mean"]
    variance: Annotated[np.ndarray, "Estimate of variance"]
    shift: np.ndarray
    scale: np.ndarray

    epsilon: float = 2 ** -52

    @classmethod
    def no_op(cls, dim: int):
        return cls(
            mean=np.zeros(dim),
            variance=np.ones(dim),
            shift=np.zeros(dim),
            scale=np.ones(dim),
        )

    def _normalize(self, data, mean, variance):
        return (data - mean) / (variance + self.epsilon) ** 0.5

    def _unnormalize(self, data):
        return data * self.scale + self.shift


@dataclass(frozen=False)
class TrainBatchNorm(BatchNorm):
    persistence: float = 0.9

    @dataclass(frozen=True)
    class Gradient:
        shift: np.ndarray
        scale: np.ndarray

    def forward(self, input: np.ndarray):
        current_mean = np.mean(input, axis=0)
        current_variance = np.var(input, axis=0)
        self.mean = self._interpolate(self.mean, current_mean)
        self.variance = self._interpolate(self.variance, current_variance)
        normalized = self._normalize(
            input, mean=current_mean, variance=current_variance
        )
        return self._unnormalize(normalized)

    def backward(self, gradient: np.ndarray, input: np.ndarray, **kwargs):
        current_mean = np.mean(input, axis=0)
        current_variance = np.var(input, axis=0)
        variance_multiplier = (current_variance + self.epsilon) ** -0.5
        normalized = self._normalize(
            input, mean=current_mean, variance=current_variance
        )
        return (
            self.Gradient(
                shift=np.sum(gradient, axis=0),
                scale=np.sum(gradient * normalized, axis=0),
            ),
            gradient * self.scale * variance_multiplier,
        )

    @property
    def test(self):
        return TestBatchNorm(
            mean=self.mean,
            variance=self.variance,
            shift=self.shift,
            scale=self.scale,
            epsilon=self.epsilon,
        )

    def _interpolate(self, estimate, current):
        return current + (estimate - current) * self.persistence


@dataclass(frozen=True)
class TestBatchNorm(BatchNorm):
    def forward(self, input: np.ndarray):
        normalized = self._normalize(input, mean=self.mean, variance=self.variance)
        return self._unnormalize(normalized)
