from typing import Dict
from dataclasses import dataclass, field
from collections import defaultdict
from contextlib import AbstractContextManager
import numpy as np
from crater.tensor import Tensor
from .active import active_recorders


@dataclass(frozen=True)
class Recorder(AbstractContextManager):
    tensor_sources: dict = field(
        default_factory=dict
    )  # mapping from tensor id to the operation which created it

    def register(self, operation):
        self.tensor_sources[operation.output.id] = operation

    def gradients(self, scalar: Tensor) -> Dict[int, np.ndarray]:
        assert all(dim == 1 for dim in scalar.shape)
        if scalar.id not in self.tensor_sources:
            raise ValueError("Cannot find gradients of untracked tensor")
        return self._gradients(
            tensor_id=scalar.id,
            outgoing_gradient=np.array(1, dtype=np.float32),
        )

    def _gradients(
        self, tensor_id: int, outgoing_gradient: np.ndarray
    ) -> Dict[int, np.ndarray]:
        result = defaultdict(lambda: 0)
        result[tensor_id] += outgoing_gradient
        if tensor_id not in self.tensor_sources:
            return result
        for connected_tensor_id, connected_gradient in (
            self.tensor_sources[tensor_id].gradients(outgoing_gradient).items()
        ):
            incoming_gradients = self._gradients(
                tensor_id=connected_tensor_id, outgoing_gradient=connected_gradient
            )
            for incoming_id, incoming_gradient in incoming_gradients.items():
                result[incoming_id] += incoming_gradient
        return dict(result)

    def __enter__(self):
        active_recorders.append(self)
        return self

    def __exit__(self, exc_type, exc_value, trace):
        active_recorders.remove(self)
