from typing import List
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Passage:
    context: np.ndarray
    targets: np.ndarray
