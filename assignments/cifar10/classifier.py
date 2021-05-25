from typing import List, Callable
from dataclasses import dataclass
import numpy as np
from .stack import Stack
from .layers import Layer, PureLayer, BiasedLayer
from .relu import ReLU
from .softmax import Softmax
from .batch_norm import TrainBatchNorm


@dataclass(frozen=True)
class Classifier(Stack):
    normalize: Callable[[np.ndarray], np.ndarray]

    @classmethod
    def from_dims(
        cls,
        dims: List[int],
        make_hidden_layer,
        make_final_layer,
        normalize,
    ):
        return cls(
            steps=tuple(
                [
                    make_hidden_layer(in_dim, out_dim)
                    for in_dim, out_dim in zip(dims[:-2], dims[1:-1])
                ]
                + [make_final_layer(dims[-2], dims[-1])]
            ),
            normalize=normalize,
        )

    @classmethod
    def layer_maker(cls, activation, batch_norm: bool):
        if batch_norm:
            return lambda in_dim, out_dim: Stack(
                steps=[
                    PureLayer.xavier(in_dim, out_dim),
                    TrainBatchNorm.no_op(out_dim),
                    activation(),
                ]
            )
        return lambda in_dim, out_dim: Stack(
            steps=[
                BiasedLayer.xavier(in_dim, out_dim),
                activation(),
            ]
        )

    def loss(self, outputs: np.ndarray, targets: List[int]):
        """Does not include regularization"""
        return -np.log(outputs[np.arange(outputs.shape[0]), targets]).mean()

    def loss_backward(self, outputs: np.ndarray, targets: List[int]):
        def single_gradient(distribution: np.ndarray, target: int):
            result = np.zeros_like(distribution)
            result[target] = -1 / distribution[target] / len(outputs)
            return result

        return np.array(
            [
                single_gradient(distribution, target)
                for distribution, target in zip(outputs, targets)
            ]
        )

    def gradient(self, input: np.ndarray, targets: List[int], **kwargs):
        outputs = self.forward(input)
        self_gradient, input_gradient = self.backward(
            self.loss_backward(outputs, targets), input=input, **kwargs
        )
        return self_gradient

    def cost(self, outputs: np.ndarray, targets: List[int], regularization: float):
        """Loss + regularization penalty"""
        return self.loss(outputs, targets) + regularization * sum(
            np.sum(step.weights ** 2)
            for layer in self.steps
            for step in layer.steps
            if isinstance(step, Layer)
        )

    def _full_forward(self, input: np.ndarray):
        normalized_input = self.normalize(input)
        return super()._full_forward(normalized_input)
