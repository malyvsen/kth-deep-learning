from typing import Dict
from dataclasses import dataclass
from .ada_grad import AdaGrad
from .rnn import RNN
from .text import Text
from .synthesize import synthesize
import plotly.graph_objects as go
from tqdm.auto import trange


@dataclass(frozen=True)
class Experiment:
    ada_grad: AdaGrad
    samples: Dict[int, str]
    losses: Dict[int, str]

    @classmethod
    def run(cls, text: Text, hidden_dim=100, num_steps=100_000, passage_length=25):
        ada_grad = AdaGrad.from_network(
            RNN.from_dims(num_classes=text.num_unique_characters, hidden_dim=hidden_dim)
        )
        samples = {}
        losses = {}

        make_passages = lambda: text.passages(length=passage_length)
        passages = make_passages()
        last_state = ada_grad.network.initial_state
        for step in trange(num_steps):
            if step % 10_000 == 0:
                samples[step] = synthesize(
                    ada_grad.network, training_text=text, length=200
                )

            try:
                passage = next(passages)
            except StopIteration:
                passages = make_passages()
                last_state = ada_grad.network.initial_state
                passage = next(passages)

            states, outputs = ada_grad.network.run(
                initial_state=last_state, passage=passage
            )
            losses[step] = ada_grad.network.loss(
                outputs=outputs, targets=passage.targets
            )
            gradients = ada_grad.network.run_backward(
                states=states, outputs=outputs, passage=passage
            )
            ada_grad = ada_grad.step(gradients)
            last_state = states[-1]

        return cls(ada_grad=ada_grad, samples=samples, losses=losses)

    @property
    def rnn(self):
        return self.ada_grad.network

    @property
    def loss_plot(self):
        smoothed_loss = [self.losses[0]]
        for loss in self.losses.values():
            smoothed_loss.append(smoothed_loss[-1] * 0.999 + loss * 0.001)
        return go.Figure(
            layout=dict(
                title="Training progress",
                xaxis_title="Number of training steps",
                yaxis_title="Smoothed loss",
            ),
            data=[
                go.Scatter(
                    y=smoothed_loss[1:],
                )
            ],
        )
