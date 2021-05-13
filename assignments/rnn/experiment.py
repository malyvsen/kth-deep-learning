from typing import Dict
from dataclasses import dataclass
import itertools
import plotly.graph_objects as go
from tqdm.auto import trange
from .ada_grad import AdaGrad
from .rnn import RNN
from .tweets import Tweets
from .synthesize import synthesize


@dataclass(frozen=True)
class Experiment:
    ada_grad: AdaGrad
    samples: Dict[int, str]
    losses: Dict[int, str]

    @classmethod
    def run(cls, tweets: Tweets, hidden_dim=100, num_steps=100_000):
        ada_grad = AdaGrad.from_network(
            RNN.from_dims(
                num_classes=tweets.num_unique_characters, hidden_dim=hidden_dim
            )
        )
        samples = {}
        losses = {}

        last_state = ada_grad.network.initial_state
        for step, tweet in zip(trange(num_steps), itertools.cycle(tweets)):
            if step % 1000 == 0:
                samples[step] = synthesize(
                    ada_grad.network, training_tweets=tweets, length=200
                )

            states, outputs = ada_grad.network.run(
                initial_state=last_state, tweet=tweet
            )
            losses[step] = ada_grad.network.loss(outputs=outputs, targets=tweet.targets)
            gradients = ada_grad.network.run_backward(
                states=states, outputs=outputs, tweet=tweet
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
