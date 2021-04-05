from typing import Dict
from dataclasses import dataclass
import numpy as np
from crater.tensor import Tensor


@dataclass(frozen=True)
class Gradient:
    tensor: Tensor
    gradient: np.ndarray

    def __post_init__(self):
        assert self.tensor.shape == self.gradient.shape
