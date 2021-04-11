from typing import Union, List, Callable
from dataclasses import dataclass
from tqdm.auto import trange
import plotly.graph_objects as go
from .data import vector_to_image
from .classifier import Classifier


@dataclass(frozen=True)
class Experiment:
    classifier: Classifier
    losses: List[float]
    accuracies: List[float]

    @classmethod
    def run(
        cls,
        data,
        classifier: Classifier,
        num_batches: int = 256,
        num_per_batch: int = 64,
        learning_rate: Union[float, Callable[[int], float]] = 1e-3,
        regularization: float = 1e-3,
    ):
        losses = []
        accuracies = []

        for batch_idx in trange(num_batches):
            num_examples_seen = batch_idx * num_per_batch
            start_idx = num_examples_seen % len(data["train"]["features"])
            batch = {
                name: array[start_idx : start_idx + num_per_batch]
                for name, array in data["train"].items()
            }
            classifier = classifier.train_step(
                batch,
                regularization=regularization,
                learning_rate=(
                    learning_rate(num_examples_seen)
                    if callable(learning_rate)
                    else learning_rate
                ),
            )
            losses.append(
                {
                    split_name: classifier.loss(
                        split, regularization=regularization
                    ).data
                    for split_name, split in data.items()
                }
            )
            accuracies.append(
                {
                    split_name: classifier.accuracy(split)
                    for split_name, split in data.items()
                }
            )

        return cls(classifier=classifier, losses=losses, accuracies=accuracies)

    @property
    def loss_plot(self):
        return go.Figure(
            layout=dict(
                title="Training progress - loss",
                xaxis_title="Batch number",
                yaxis_title="Loss",
            ),
            data=[
                go.Scatter(
                    name=split_name.title(),
                    y=[loss[split_name] for loss in self.losses],
                )
                for split_name in self.losses[0].keys()
            ],
        )

    @property
    def accuracy_plot(self):
        return go.Figure(
            layout=dict(
                title="Training progress - accuracy",
                xaxis_title="Batch number",
                yaxis_title="Accuracy",
            ),
            data=[
                go.Scatter(
                    name=split_name.title(),
                    y=[accuracy[split_name] for accuracy in self.accuracies],
                )
                for split_name in self.accuracies[0].keys()
            ],
        )


def train_epoch(train_data, classifier, num_per_batch, learning_rate, regularization):
    for start_idx in range(0, len(train_data["features"]), num_per_batch):
        batch = {
            name: array[start_idx : start_idx + num_per_batch]
            for name, array in train_data.items()
        }
        classifier.train_step(
            batch, regularization=regularization, learning_rate=learning_rate
        )
