from typing import Annotated, Tuple
from dataclasses import dataclass
from enum import Enum
from contextlib import contextmanager
import numpy as np
from crater.tensor import Tensor
from crater.operations import coalesce


@dataclass
class BatchNormalization:
    mean: Annotated[np.ndarray, "Estimate of mean"]
    variance: Annotated[np.ndarray, "Estimate of variance"]
    persistence: float
    epsilon = 2 ** -52

    shift: Tensor
    scale: Tensor

    class Mode(Enum):
        train = "train"
        test = "test"
        disabled = "disabled"

    _mode = None

    @classmethod
    def from_shape(
        cls,
        shape: Tuple[int],
        persistence: float = None,
    ) -> "BatchNormalization":
        return cls(
            mean=np.zeros(shape),
            variance=np.ones(shape),
            persistence=persistence if persistence is not None else 0.9,
            shift=Tensor.from_numpy(np.zeros(shape)),
            scale=Tensor.from_numpy(np.ones(shape)),
        )

    def __call__(self, input: Tensor) -> Tensor:
        input = coalesce(input)
        if self._mode == self.Mode.train:
            current_mean = np.mean(input.numpy, axis=0)
            current_variance = np.var(input.numpy, axis=0)
            normalized = self._normalize(
                input, mean=current_mean, variance=current_variance
            )
            self.mean = self._interpolate(self.mean, current_mean)
            self.variance = self._interpolate(self.variance, current_variance)
        elif self._mode == self.Mode.test:
            normalized = self._normalize(input, mean=self.mean, variance=self.variance)
        elif self._mode == self.Mode.disabled:
            return input
        else:
            raise ValueError(
                "BatchNormalization must be called inside a BatchNormalization.mode context"
            )
        return normalized * self.scale + self.shift

    @classmethod
    @contextmanager
    def mode(cls, mode: Mode):
        old_mode = cls._mode
        cls._mode = cls.Mode(mode)
        yield
        cls._mode = old_mode

    def _interpolate(self, estimate, current):
        return current + (estimate - current) * self.persistence

    def _normalize(self, data, mean, variance):
        return (data - mean) / (variance + self.epsilon) ** 0.5
