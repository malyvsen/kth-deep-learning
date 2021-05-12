from typing import Dict
from dataclasses import dataclass, asdict, replace
import numpy as np
from crater import Tensor, Gradients


@dataclass(frozen=True)
class AdaGrad:
    network: object
    magnitudes: Dict[str, np.ndarray]
    learning_rate: float
    epsilon: float = 2 ** -32

    @classmethod
    def from_network(cls, network, learning_rate=0.1):
        dict_form = asdict(network)
        return cls(
            network=network,
            magnitudes={
                name: np.zeros(param.shape)
                for name, param in dict_form.items()
                if isinstance(param, Tensor)
            },
            learning_rate=learning_rate,
        )

    def step(self, gradients: Gradients) -> "AdaGrad":
        """Perform a single optimizer step. Returns the new optimizer with a new network inside."""
        magnitudes = {
            name: magnitude + gradients[getattr(self.network, name)] ** 2
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
                        * gradients[getattr(self.network, name)]
                    ).no_backward
                    for name, magnitude in magnitudes.items()
                }
            ),
            magnitudes=magnitudes,
            learning_rate=self.learning_rate,
        )
