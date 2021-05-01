from typing import Dict
from dataclasses import dataclass
import numpy as np
from tqdm.auto import trange
import plotly.graph_objects as go
from crater.premade import Classifier, CyclicLearningRate, BatchNormalization
from .data import vector_to_image


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
        batch_norm_mode: BatchNormalization.Mode,
        num_cycles: int = 1,
        measurements_per_cycle: int = 10,
        batch_size: int = 100,
        regularization: float = 1e-3,
    ):
        losses = {}
        accuracies = {}

        def evaluate():
            with BatchNormalization.mode("test"):
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

        for batch_idx in trange(learning_rate.batches_per_cycle * num_cycles):
            if batch_idx % (len(data["train"]["features"]) // batch_size) == 0:
                permutation = np.random.permutation(len(data["train"]["features"]))
                permuted = {
                    name: array[permutation] for name, array in data["train"].items()
                }
            start_idx = (batch_idx * batch_size) % len(data["train"]["features"])
            batch = {
                name: array[start_idx : start_idx + batch_size]
                for name, array in permuted.items()
            }
            with BatchNormalization.mode(batch_norm_mode):
                classifier = classifier.train_step(
                    batch,
                    regularization=regularization,
                    learning_rate=learning_rate.learning_rate(batch_idx),
                )
            if (
                measurements_per_cycle > 0
                and batch_idx
                % (learning_rate.batches_per_cycle // measurements_per_cycle)
                == 0
            ):
                evaluate()

        evaluate()
        return cls(classifier=classifier, losses=losses, accuracies=accuracies)

    @property
    def final_accuracy(self):
        return list(self.accuracies.values())[-1]["validation"]

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
