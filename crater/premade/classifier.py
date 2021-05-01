from typing import List, Callable
from dataclasses import dataclass, replace
from functools import reduce
import numpy as np
from crater.tensor import Tensor
from crater.gradient import Gradients, Gradient
from crater.utils import one_hot
from .layer import Layer
from .batch_normalization import BatchNormalization


@dataclass(frozen=True)
class Classifier:
    layers: List[Layer]
    batch_normalizations: List[BatchNormalization]
    normalize: Callable[[np.ndarray], Tensor]

    @classmethod
    def from_dims(cls, dims, normalize, persistence: float = None) -> "Classifier":
        batch_normalizations = [
            BatchNormalization.from_shape(dim, persistence=persistence)
            for dim in dims[1:]
        ]
        return cls(
            layers=[
                Layer.from_dims(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    activation=cls._make_activation(batch_norm)
                    if layer_idx < len(dims) - 2
                    else batch_norm,
                )
                for layer_idx, (in_dim, out_dim, batch_norm) in enumerate(
                    zip(
                        dims[:-1],
                        dims[1:],
                        batch_normalizations,
                    )
                )
            ],
            batch_normalizations=batch_normalizations,
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
        batch_normalizations = (
            self.batch_normalizations
            if BatchNormalization._mode == BatchNormalization.Mode.disabled
            else [
                replace(
                    batch_norm,
                    shift=(
                        batch_norm.shift - gradients[batch_norm.shift] * learning_rate
                    ).no_backward,
                    scale=(
                        batch_norm.scale - gradients[batch_norm.scale] * learning_rate
                    ).no_backward,
                )
                for batch_norm in self.batch_normalizations
            ]
        )
        return replace(
            self,
            layers=[
                Layer(
                    weights=(
                        layer.weights - gradients[layer.weights] * learning_rate
                    ).no_backward,
                    biases=(
                        layer.biases - gradients[layer.biases] * learning_rate
                    ).no_backward,
                    activation=self._make_activation(batch_norm),
                )
                for layer, batch_norm in zip(self.layers, batch_normalizations)
            ],
            batch_normalizations=batch_normalizations,
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

    @classmethod
    def _make_activation(cls, batch_norm):
        return lambda tensor: batch_norm(tensor).clip(low=0)
