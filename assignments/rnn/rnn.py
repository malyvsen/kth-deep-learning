from typing import List
from dataclasses import dataclass, asdict
import numpy as np
from .operations import (
    one_hot,
    matmul_backward,
    tanh,
    tanh_backward,
    softmax,
    softmax_backward,
)
from .passage import Passage


@dataclass(frozen=True)
class RNN:
    hidden_weights: np.ndarray
    input_weights: np.ndarray
    hidden_biases: np.ndarray
    output_weights: np.ndarray
    output_biases: np.ndarray

    @classmethod
    def from_dims(cls, num_classes: int, hidden_dim: int) -> "RNN":
        return cls(
            hidden_weights=np.random.normal(
                scale=hidden_dim ** -0.5, size=(hidden_dim, hidden_dim)
            ),
            input_weights=np.random.normal(
                scale=num_classes ** -0.5, size=(num_classes, hidden_dim)
            ),
            hidden_biases=np.zeros(hidden_dim),
            output_weights=np.random.normal(
                scale=hidden_dim ** -0.5, size=(hidden_dim, num_classes)
            ),
            output_biases=np.zeros(num_classes),
        )

    @property
    def num_classes(self) -> int:
        return self.output_biases.shape[0]

    @property
    def initial_state(self) -> np.ndarray:
        return np.zeros_like(self.hidden_biases)

    def run(self, initial_state: np.ndarray, passage: Passage):
        """
        Perform several steps.
        Takes:
        * state (state_size,)
        * passage (Passage object)
        Returns:
        * states (sequence_length + 1, state_size)
        * outputs (sequence_length, num_classes)
        """
        states = [initial_state]
        outputs = []
        for input in passage.context:
            state, output = self.step(
                [states[-1]],
                input=[input],
                new_state_gradient=None,
                output_gradient=None,
            )
            states.append(state[0])
            outputs.append(output[0])
        return states, outputs

    def run_backward(
        self, states: List[np.ndarray], outputs: List[np.ndarray], passage: Passage
    ):
        result = {key: 0 for key in asdict(self)}
        output_gradients = self.loss_backward(outputs=outputs, targets=passage.targets)
        state_gradient = 0
        for input, prev_state, output_gradient in zip(
            passage.context[::-1], states[-2::-1], output_gradients[::-1]
        ):
            state_gradient, param_gradient = self.step(
                state=np.array([prev_state]),
                input=np.array([input]),
                new_state_gradient=state_gradient,
                output_gradient=output_gradient,
            )
            result = {
                key: value + getattr(param_gradient, key)
                for key, value in result.items()
            }
        return result

    def step(
        self,
        state: np.ndarray,
        input: np.ndarray,
        new_state_gradient: np.ndarray,
        output_gradient: np.ndarray,
    ):
        """
        Perform a single step (not recurrent).
        Takes:
        * state (batch_size, state_size)
        * input (batch_size,)
        * gradient wrt new state (batch_size, state_size)
        * gradient wrt output (batch_size, num_classes)
        Returns:
        * new state (batch_size, state_size)
        * output (batch_size, num_classes)
        Or, if gradients are provided:
        * gradient wrt state (batch_size, state_size)
        * gradient wrt parameters (RNN object)
        """
        pre_tanh = (
            np.matmul(state, self.hidden_weights)
            + np.matmul(
                one_hot(input, num_classes=self.num_classes), self.input_weights
            )
            + self.hidden_biases
        )
        new_state = tanh(pre_tanh)
        pre_softmax = np.matmul(new_state, self.output_weights) + self.output_biases
        output = softmax(pre_softmax, axis=-1)

        if new_state_gradient is None and output_gradient is None:
            return new_state, output

        pre_softmax_gradient = softmax_backward(
            output_gradient, input=pre_softmax, axis=-1
        )
        new_state_gradient_partial, output_weights_gradient = matmul_backward(
            pre_softmax_gradient, left=new_state, right=self.output_weights
        )
        pre_tanh_gradient = tanh_backward(
            new_state_gradient + new_state_gradient_partial, input=pre_tanh
        )
        _, input_weights_gradient = matmul_backward(
            pre_tanh_gradient,
            left=one_hot(input, num_classes=self.num_classes),
            right=self.input_weights,
        )
        state_gradient, hidden_weights_gradient = matmul_backward(
            pre_tanh_gradient, left=state, right=self.hidden_weights
        )
        return (
            state_gradient,
            type(self)(
                hidden_weights=hidden_weights_gradient,
                input_weights=input_weights_gradient,
                hidden_biases=pre_tanh_gradient[0],
                output_weights=output_weights_gradient,
                output_biases=pre_softmax_gradient[0],
            ),
        )

    def loss(self, outputs: List[np.ndarray], targets: List[int]):
        return -sum(
            np.log(distribution[target])
            for distribution, target in zip(outputs, targets)
        )

    def loss_backward(self, outputs: List[np.ndarray], targets: List[int]):
        def single_gradient(distribution: np.ndarray, target: int):
            result = np.zeros_like(distribution)
            result[target] = -1 / distribution[target]
            return result

        return [
            single_gradient(distribution, target)
            for distribution, target in zip(outputs, targets)
        ]
