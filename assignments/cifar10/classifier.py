from typing import List
from dataclasses import dataclass
from .stack import Stack
from .layers import PureLayer, BiasedLayer
from .relu import ReLU
from .batch_norm import TrainBatchNorm


@dataclass(frozen=True)
class Classifier(Stack):
    @classmethod
    def from_dims(
        cls,
        dims: List[int],
        make_hidden_layer,
        make_final_layer=BiasedLayer.xavier,
    ):
        return cls(
            steps=[
                make_hidden_layer(in_dim, out_dim)
                for in_dim, out_dim in zip(dims[:-2], dims[1:-1])
            ]
            + [make_final_layer(dims[-2], dims[-1])]
        )

    @classmethod
    def hidden_layer_maker(cls, batch_norm=True):
        if batch_norm:
            return lambda in_dim, out_dim: Stack(
                steps=[
                    PureLayer.xavier(in_dim, out_dim),
                    TrainBatchNorm.no_op(out_dim),
                    ReLU(),
                ]
            )
        return lambda in_dim, out_dim: Stack(
            steps=[
                BiasedLayer.xavier(in_dim, out_dim),
                ReLU(),
            ]
        )
