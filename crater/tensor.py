from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Tensor:
    data: np.ndarray

    def __init__(self, data: np.ndarray):
        assert data.flags.writeable == False
        self.__dict__.update(dict(data=data))

    @classmethod
    def from_numpy(cls, data: np.ndarray):
        if data.flags.writeable:
            data = data.copy()
            data.flags.writeable = False
        return cls(data=data)

    @classmethod
    def from_builtin(cls, data):
        return cls.from_numpy(np.array(data))

    @property
    def shape(self):
        return self.data.shape

    def __eq__(self, other):
        return np.all(self.data == other.data)

    @property
    def id(self) -> int:
        return id(self)

    __hash__ = None
