from typing import Union, Callable
from dataclasses import dataclass, replace
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

    @classmethod
    def convert(cls, thing):
        """Convert anything to a tensor, if not already a tensor"""
        if isinstance(thing, cls):
            return thing
        if isinstance(thing, np.ndarray):
            return cls.from_numpy(thing)
        return cls.from_builtin(thing)

    @property
    def shape(self):
        return self.data.shape

    @property
    def numpy(self):
        return self.data.copy()

    @property
    def no_backward(self):
        return replace(self, backward=None)

    def __eq__(self, other):
        return np.all(self.data == other.data) and self.shape == other.shape

    __hash__ = None
