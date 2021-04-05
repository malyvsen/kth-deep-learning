from crater import Tensor, Gradients
from crater.operations import matrix_multiply


def test_train_step():
    data = Tensor.from_builtin([[1, 2, 3], [4, 5, 0]])
    targets = Tensor.from_builtin([[1, 0], [0, 1]])
    weights = Tensor.from_builtin([[2, 0], [1, 1], [0, 3]])
    biases = Tensor.from_builtin([-2, -7])

    logits = matrix_multiply(data, weights) + biases
    probabilities = logits.softmax(-1)
    loss = -(probabilities.log() * targets).sum()
    gradients = Gradients.trace(loss)

    weights -= Tensor.from_numpy(gradients[weights])
    biases -= Tensor.from_numpy(gradients[biases])
