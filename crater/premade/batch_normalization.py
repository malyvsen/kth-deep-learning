from typing import Annotated, Tuple
from dataclasses import dataclass
from enum import Enum
from contextlib import contextmanager
import numpy as np
from crater.tensor import Tensor


@dataclass
class BatchNormalization:
    mean: Annotated[np.ndarray, "Estimate of mean"]
    standard_deviation: Annotated[np.ndarray, "Estimate of standard deviation"]
    persistence: float

    shift: Tensor
    scale: Tensor

    class Mode(Enum):
        train = "train"
        test = "test"

    _mode = None

    @classmethod
    def from_shape(
        cls, shape: Tuple[int], persistence: float = None
    ) -> "BatchNormalization":
        return cls(
            mean=np.random.normal(shape),
            standard_deviation=np.random.lognormal(shape),
            persistence=persistence if persistence is not None else 0.9,
            shift=Tensor.from_numpy(np.random.normal(shape)),
            scale=Tensor.from_numpy(np.random.normal(shape)),
        )

    def __call__(self, input: Tensor) -> Tensor:
        if self._mode == Mode.train:
            current_mean = np.mean(input.numpy, axis=0)
            current_standard_deviation = np.std(input.numpy, axis=0)
            normalized = (input - current_mean) / current_standard_deviation
            self.mean = self._interpolate(self.mean, current_mean)
            self.standard_deviation = self._interpolate(
                self.standard_deviation, current_standard_deviation
            )
        elif self._mode == Mode.test:
            normalized = (input - self.mean) / self.standard_deviation
        else:
            raise ValueError(
                "BatchNormalization must be called inside a BatchNormalization.mode context"
            )
        return normalized * self.scale + self.shift

    @contextmanager
    @classmethod
    def mode(cls, mode: Mode):
        old_mode = cls._mode
        cls._mode = mode
        yield
        cls._mode = old_mode

    def _interpolate(self, estimate, current):
        return current + (estimate - current) * self.persistence
