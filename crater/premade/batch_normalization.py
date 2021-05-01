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
        cls,
        shape: Tuple[int],
        persistence: float = None,
    ) -> "BatchNormalization":
        return cls(
            mean=np.zeros(shape),
            standard_deviation=np.ones(shape),
            persistence=persistence if persistence is not None else 0.9,
            shift=Tensor.from_numpy(np.zeros(shape)),
            scale=Tensor.from_numpy(np.ones(shape)),
        )

    def __call__(self, input: Tensor) -> Tensor:
        input = coalesce(input)
        if self._mode == self.Mode.train:
            current_mean = np.mean(input.numpy, axis=0)
            current_standard_deviation = np.std(input.numpy, axis=0)
            normalized = (input - current_mean) / current_standard_deviation
            self.mean = self._interpolate(self.mean, current_mean)
            self.standard_deviation = self._interpolate(
                self.standard_deviation, current_standard_deviation
            )
        elif self._mode == self.Mode.test:
            normalized = (input - self.mean) / self.standard_deviation
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
