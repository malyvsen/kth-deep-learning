from typing import List, Callable
from dataclasses import dataclass, replace
from functools import reduce
import numpy as np
from crater import Tensor, Gradients, Layer
from crater.operations import matrix_multiply
from crater.utils import one_hot


@dataclass(frozen=True)
class Classifier:
    layers: List[Layer]
    normalize: Callable[[np.ndarray], Tensor]

    @classmethod
    def from_dims(cls, hidden_dims, normalize):
        return cls(
            layers=[
                Layer.from_dims(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    activation=lambda tensor: tensor.clip(low=0),
                )
                for in_dim, out_dim in zip(
                    [3 * 32 * 32] + hidden_dims[:-1],
                    hidden_dims,
                )
            ]
            + [
                Layer.from_dims(
                    in_dim=hidden_dims[-1], out_dim=10, activation=lambda tensor: tensor
                )
            ],
            normalize=normalize,
        )

    def logits(self, data):
        return reduce(lambda tensor, layer: layer(tensor), self.layers, data)

    def probabilities(self, data):
        return self.logits(data).softmax(-1)

    def loss(self, batch, regularization):
        cross_entropy = -(
            (
                self.probabilities(batch["features"]).log()
                * one_hot(batch["labels"], num_classes=10)
            ).sum()
            / Tensor.from_builtin(len(batch["features"]))
        )
        penalty = (
            sum((layer.weights * layer.weights).sum() for layer in self.layers)
            * regularization
        )
        return cross_entropy + penalty

    def gradients(self, batch, regularization):
        return Gradients.trace(self.loss(batch, regularization))

    def predictions(self, data):
        return np.argmax(self.logits(data).data, axis=-1)

    def accuracy(self, batch):
        return np.mean(self.predictions(batch["features"]) == batch["labels"])

    def train_step(self, batch, regularization, learning_rate):
        gradients = self.gradients(batch, regularization)
        return replace(
            self,
            layers=[
                replace(
                    layer,
                    weights=Tensor.from_numpy(
                        layer.weights.data - gradients[layer.weights] * learning_rate
                    ),
                    biases=Tensor.from_numpy(
                        layer.biases.data - gradients[layer.biases] * learning_rate
                    ),
                )
                for layer in self.layers
            ],
        )
