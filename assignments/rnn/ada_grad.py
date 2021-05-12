from typing import Dict
from dataclasses import dataclass, asdict, replace
import numpy as np
from .rnn import RNN


@dataclass(frozen=True)
class AdaGrad:
    network: RNN
    magnitudes: Dict[str, np.ndarray]
    learning_rate: float
    epsilon: float = 2 ** -32

    @classmethod
    def from_network(cls, network, learning_rate=0.1):
        return cls(
            network=network,
            magnitudes={
                name: np.zeros(param.shape) for name, param in asdict(network).items()
            },
            learning_rate=learning_rate,
        )

    def step(self, gradients: Dict[str, np.ndarray]) -> "AdaGrad":
        """Perform a single optimizer step. Returns the new optimizer with a new network inside."""
        magnitudes = {
            name: magnitude + gradients[name] ** 2
            for name, magnitude in self.magnitudes.items()
        }
        return replace(
            self,
            network=type(self.network)(
                **{
                    name: (
                        getattr(self.network, name)
                        - self.learning_rate
                        / (magnitude + self.epsilon) ** 0.5
                        * gradients[name]
                    )
                    for name, magnitude in magnitudes.items()
                }
            ),
            magnitudes=magnitudes,
            learning_rate=self.learning_rate,
        )
