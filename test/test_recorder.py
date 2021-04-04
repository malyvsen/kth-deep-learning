import pytest
import numpy as np
from crater import Tensor, Recorder
from crater.operations import Operation


def test_recording():
    tensor = Tensor.from_builtin([1, 2, 3])

    class TestOperation(Operation):
        def forward(self, input):
            return Tensor.from_builtin(input.data.sum())

        def backward(self, outgoing_gradient):
            return {self.input.id: np.ones_like(self.input.data)}

    with Recorder() as recorder:
        output = TestOperation.apply(tensor)
    gradients = recorder.gradients(output)

    assert gradients.keys() == {tensor.id, output.id}
    assert np.allclose(gradients[tensor.id], np.array([1, 1, 1]))
    assert np.allclose(gradients[output.id], 1)


def test_stop_recording():
    tensor = Tensor.from_builtin([1, 2, 3])

    class TestOperation(Operation):
        def forward(self, input):
            return Tensor.from_builtin(input.data.sum())

        def backward(self, outgoing_gradient):
            return {self.input.id: np.ones_like(self.input.data)}

    with Recorder() as recorder:
        output = TestOperation.apply(tensor)

    untracked = TestOperation.apply(output)
    with pytest.raises(ValueError):
        recorder.gradients(untracked)
