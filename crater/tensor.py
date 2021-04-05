from typing import Union, Callable
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Tensor:
    data: np.ndarray
    backward: Union[None, Callable]

    def __post_init__(self):
        assert self.data.flags.writeable == False

    @classmethod
    def from_numpy(cls, data: np.ndarray, backward=None):
        if data.flags.writeable:
            data = data.copy()
            data.flags.writeable = False
        return cls(data=data, backward=backward)

    @classmethod
    def from_builtin(cls, data, backward=None):
        return cls.from_numpy(np.array(data), backward=backward)

    @property
    def shape(self):
        return self.data.shape

    def __eq__(self, other):
        return np.all(self.data == other.data)

    __hash__ = None
