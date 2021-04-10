import numpy as np
from crater import Tensor, Gradients
from crater.operations import matrix_multiply
from crater.utils import one_hot


class Classifier:
    def __init__(self, normalize):
        self.normalize = normalize
        self.weights = Tensor.from_numpy(
            np.random.normal(scale=0.01, size=[32 * 32 * 3, 10])
        )
        self.biases = Tensor.from_numpy(np.random.normal(scale=0.01, size=[10]))

    def logits(self, data):
        return matrix_multiply(self.normalize(data), self.weights) + self.biases

    def probabilities(self, data):
        return self.logits(data).softmax(-1)

    def loss(self, batch, regularization):
        cross_entropy = -(
            (
                self.probabilities(batch["data"]).log()
                * one_hot(batch["labels"], num_classes=10)
            ).sum()
            / Tensor.from_builtin(len(batch["data"]))
        )
        penalty = (self.weights * self.weights).sum() * Tensor.from_builtin(
            regularization
        )
        return cross_entropy + penalty

    def gradients(self, batch, regularization):
        return Gradients.trace(self.loss(batch, regularization))

    def predictions(self, data):
        return np.argmax(self.logits(data).data, axis=-1)

    def accuracy(self, batch):
        return np.mean(self.predictions(batch["data"]) == batch["labels"])

    def train_step(self, batch, regularization, learning_rate):
        gradients = self.gradients(batch, regularization)
        self.weights = Tensor.from_numpy(
            self.weights.data - gradients[self.weights] * learning_rate
        )
        self.biases = Tensor.from_numpy(
            self.biases.data - gradients[self.biases] * learning_rate
        )
