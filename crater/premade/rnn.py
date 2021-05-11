from typing import List
from dataclasses import dataclass
import numpy as np
from crater.tensor import Tensor
from crater.utils import one_hot
from crater.operations import matrix_multiply


@dataclass(frozen=True)
class RNN:
    hidden_weights: Tensor
    input_weights: Tensor
    hidden_biases: Tensor
    output_weights: Tensor
    output_biases: Tensor

    @classmethod
    def from_dims(cls, num_classes: int, hidden_dim: int) -> "RNN":
        return cls(
            hidden_weights=Tensor.from_numpy(
                np.random.normal(
                    scale=hidden_dim ** -0.5, size=(hidden_dim, hidden_dim)
                )
            ),
            input_weights=Tensor.from_numpy(
                np.random.normal(
                    scale=num_classes ** -0.5, size=(num_classes, hidden_dim)
                )
            ),
            hidden_biases=Tensor.from_numpy(np.zeros(hidden_dim)),
            output_weights=Tensor.from_numpy(
                np.random.normal(
                    scale=hidden_dim ** -0.5, size=(hidden_dim, num_classes)
                )
            ),
            output_biases=Tensor.from_numpy(np.zeros(num_classes)),
        )

    @property
    def num_classes(self):
        return self.output_biases.shape[0]

    @property
    def initial_state(self) -> Tensor:
        return Tensor.from_numpy(np.zeros(self.hidden_biases.shape))

    def run(self, initial_state: Tensor, sequences: List[np.ndarray]):
        """
        Perform several steps.
        State shape: (batch_size, state_size)
        Sequences shape: (sequence_length, batch_size)
        Returns new states (sequence_length, batch_size, state_size) and outputs (sequence_length, batch_size, num_classes)
        """
        states = []
        outputs = []
        for input in sequences:
            state, output = self.step(
                states[-1] if len(states) > 0 else initial_state,
                input=input,
            )
            states.append(state)
            outputs.append(output)
        return states, outputs

    def step(self, state: Tensor, input: np.ndarray):
        """
        Perform a single step (not recurrent).
        State shape: (batch_size, state_size)
        Input shape: (batch_size,)
        Returns new state (batch_size, state_size) and output (batch_size, output_size)
        """
        new_state = (
            matrix_multiply(state, self.hidden_weights)
            + matrix_multiply(
                one_hot(input, num_classes=self.num_classes), self.input_weights
            )
            + self.hidden_biases
        ).tanh()
        output = (
            matrix_multiply(new_state, self.output_weights) + self.output_biases
        ).softmax(-1)
        return new_state, output

    def loss(self, predictions: List[Tensor], targets: List[np.ndarray]):
        return -sum(
            prediction.log() * one_hot(target, num_classes=self.num_classes)
            for prediction, target in zip(predictions, targets)
        )
