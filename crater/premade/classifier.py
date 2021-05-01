from typing import List, Callable
from dataclasses import dataclass, replace
from functools import reduce
import numpy as np
from crater.tensor import Tensor
from crater.gradient import Gradients, Gradient
from crater.utils import one_hot
from .layer import Layer


@dataclass(frozen=True)
class Classifier:
    layers: List[Layer]
    normalize: Callable[[np.ndarray], Tensor]

    @classmethod
    def from_dims(cls, dims, normalize) -> "Classifier":
        return cls(
            layers=[
                Layer.from_dims(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    activation=lambda tensor: tensor.clip(low=0),
                )
                for in_dim, out_dim in zip(
                    dims[:-1],
                    dims[1:],
                )
            ],
            normalize=normalize,
        )

    @property
    def num_classes(self) -> int:
        return self.layers[-1].biases.shape[0]

    def logits(self, data) -> Tensor:
        return reduce(
            lambda tensor, layer: layer(tensor), self.layers, self.normalize(data)
        )

    def probabilities(self, data) -> Tensor:
        return self.logits(data).softmax(-1)

    def loss(self, batch, regularization) -> Tensor:
        cross_entropy = -(
            (
                self.probabilities(batch["features"]).log()
                * one_hot(batch["labels"], num_classes=self.num_classes)
            ).sum()
            / Tensor.from_builtin(len(batch["features"]))
        )
        penalty = (
            sum((layer.weights * layer.weights).sum() for layer in self.layers)
            * regularization
        )
        return cross_entropy + penalty

    def gradients(self, batch, regularization) -> Gradients:
        return Gradients.trace(self.loss(batch, regularization))

    def predictions(self, data) -> np.ndarray:
        return np.argmax(self.logits(data).data, axis=-1)

    def accuracy(self, batch) -> float:
        return np.mean(self.predictions(batch["features"]) == batch["labels"])

    def train_step(self, batch, regularization, learning_rate) -> "Classifier":
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

    @property
    def templates(self) -> np.ndarray:
        return reduce(
            lambda array, layer: np.matmul(
                array - layer.biases.data, layer.weights.data.T
            ),
            self.layers[::-1],
            one_hot(list(range(self.num_classes)), num_classes=self.num_classes).data,
        )
