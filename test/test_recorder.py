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


def test_accumulate_gradient():
    start = Tensor.from_builtin([1, 2, 3])

    class UnaryOperation(Operation):
        def forward(self, input):
            return Tensor.from_numpy(input.data * np.array([1, 1, 2]))

        def backward(self, outgoing_gradient):
            return {self.input.id: outgoing_gradient * np.array([1, 1, 2])}

    class BinaryOperation(Operation):
        def forward(self, left, right):
            return Tensor.from_builtin(np.sum(2 * left.data + right.data))

        def backward(self, outgoing_gradient):
            return {
                self.left.id: np.ones_like(self.left.data) * 2,
                self.right.id: np.ones_like(self.right.data),
            }

    with Recorder() as recorder:
        left = UnaryOperation.apply(start)
        right = UnaryOperation.apply(start)
        end = BinaryOperation.apply(left, right)

    gradients = recorder.gradients(end)

    assert gradients.keys() == {start.id, left.id, right.id, end.id}
    assert np.allclose(gradients[start.id], np.array([3, 3, 6]))
    assert np.allclose(gradients[left.id], np.array([2, 2, 2]))
    assert np.allclose(gradients[right.id], np.array([1, 1, 1]))
    assert np.allclose(gradients[end.id], np.array([1]))
