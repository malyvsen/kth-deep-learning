from typing import Dict, Any
from dataclasses import dataclass
import numpy as np
from tqdm.auto import trange
import plotly.graph_objects as go
from .classifier import Classifier
from .sgd import sgd
from .cyclic_learning_rate import CyclicLearningRate


@dataclass(frozen=True)
class Experiment:
    classifier: Classifier
    costs: Dict[int, float]
    accuracies: Dict[int, float]

    @classmethod
    def run(
        cls,
        data,
        classifier: Classifier,
        learning_rate: CyclicLearningRate,
        regularization: float,
        num_cycles: int = 1,
        measurements_per_cycle: int = 10,
        batch_size: int = 100,
    ):
        costs = {}
        accuracies = {}

        def evaluate():
            predictions = {
                split_name: classifier.test.forward(split["features"])
                for split_name, split in data.items()
            }
            costs[batch_idx] = {
                split_name: classifier.test.cost(
                    prediction, split["labels"], regularization=regularization
                )
                for split_name, split, prediction in zip(
                    data.keys(), data.values(), predictions.values()
                )
            }
            accuracies[batch_idx] = {
                split_name: (np.argmax(prediction, axis=1) == split["labels"]).mean()
                for split_name, split, prediction in zip(
                    data.keys(), data.values(), predictions.values()
                )
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
            classifier = sgd(
                classifier,
                gradient=classifier.gradient(
                    batch["features"], batch["labels"], regularization=regularization
                ),
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
        return cls(classifier=classifier, costs=costs, accuracies=accuracies)

    @property
    def final_accuracy(self):
        return list(self.accuracies.values())[-1]["validation"]

    @property
    def cost_plot(self):
        return self._plot("cost", self.costs)

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
