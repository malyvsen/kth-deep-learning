from dataclasses import asdict, replace
import numpy as np


def sgd(function, gradient, learning_rate: float):
    if isinstance(gradient, np.ndarray):
        return function - gradient * learning_rate
    return replace(
        function,
        **{
            name: sgd(
                function=getattr(function, name),
                gradient=value,
                learning_rate=learning_rate,
            )
            for name, value in asdict(gradient).items()
        }
    )
