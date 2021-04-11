from typing import Dict
from dataclasses import dataclass
from tqdm.auto import trange
import plotly.graph_objects as go
from .data import vector_to_image
from .classifier import Classifier
from .learning_rate import CyclicLearningRate


@dataclass(frozen=True)
class Experiment:
    classifier: Classifier
    losses: Dict[int, float]
    accuracies: Dict[int, float]

    @classmethod
    def run(
        cls,
        data,
        classifier: Classifier,
        learning_rate: CyclicLearningRate,
        num_cycles: int = 1,
        batches_per_measurement: int = 10,
        batch_size: int = 100,
        regularization: float = 1e-3,
    ):
        losses = {}
        accuracies = {}

        for batch_idx in trange(learning_rate.batches_per_cycle * num_cycles):
            start_idx = (batch_idx * batch_size) % len(data["train"]["features"])
            batch = {
                name: array[start_idx : start_idx + batch_size]
                for name, array in data["train"].items()
            }
            classifier = classifier.train_step(
                batch,
                regularization=regularization,
                learning_rate=learning_rate.learning_rate(batch_idx),
            )
            if batch_idx % batches_per_measurement == 0:
                losses[batch_idx] = {
                    split_name: classifier.loss(
                        split, regularization=regularization
                    ).data
                    for split_name, split in data.items()
                }
                accuracies[batch_idx] = {
                    split_name: classifier.accuracy(split)
                    for split_name, split in data.items()
                }

        return cls(classifier=classifier, losses=losses, accuracies=accuracies)

    @property
    def loss_plot(self):
        return self._plot("loss", self.losses)

    @property
    def accuracy_plot(self):
        return self._plot("accuracy", self.accuracies)

    @staticmethod
    def _plot(name: str, series: Dict[int, float]):
        return go.Figure(
            layout=dict(
                title=f"Training progress - {name}",
                xaxis_title="Batch number",
                yaxis_title=name.title(),
            ),
            data=[
                go.Scatter(
                    name=split_name.title(),
                    x=list(series.keys()),
                    y=[accuracy[split_name] for accuracy in series.values()],
                )
                for split_name in series[0].keys()
            ],
        )


def train_epoch(train_data, classifier, batch_size, learning_rate, regularization):
    for start_idx in range(0, len(train_data["features"]), batch_size):
        batch = {
            name: array[start_idx : start_idx + batch_size]
            for name, array in train_data.items()
        }
        classifier.train_step(
            batch, regularization=regularization, learning_rate=learning_rate
        )
