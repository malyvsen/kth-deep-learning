from dataclasses import dataclass, replace
import numpy as np


@dataclass(frozen=True)
class Stack:
    steps: list

    def forward(self, input: np.ndarray):
        return self._full_forward(input)[-1]

    def backward(self, gradient: np.ndarray, input: np.ndarray, **kwargs):
        output_gradient = gradient
        step_gradients = []
        for input, step in zip(
            reversed(self._full_forward(input)[:-1]), reversed(self.steps)
        ):
            step_gradient, output_gradient = step.backward(
                gradient=output_gradient, input=input, **kwargs
            )
            step_gradients.append(step_gradient)
        return Stack(steps=tuple(reversed(step_gradients))), output_gradient

    def _full_forward(self, input: np.ndarray):
        outputs = [input]
        for step in self.steps:
            outputs.append(step.forward(outputs[-1]))
        return outputs

    @property
    def test(self):
        return replace(
            self,
            steps=[step.test if hasattr(step, "test") else step for step in self.steps],
        )
