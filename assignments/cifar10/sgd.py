from collections.abc import Iterable
from dataclasses import fields, replace
import numpy as np


def sgd(function, gradient, learning_rate: float):
    if isinstance(gradient, np.ndarray):
        return function - gradient * learning_rate
    if isinstance(gradient, Iterable):
        return type(function)(
            [
                sgd(function=sub_func, gradient=sub_grad, learning_rate=learning_rate)
                for sub_func, sub_grad in zip(function, gradient)
            ]
        )
    return replace(
        function,
        **{
            field.name: sgd(
                function=getattr(function, field.name),
                gradient=getattr(gradient, field.name),
                learning_rate=learning_rate,
            )
            for field in fields(gradient)
        },
    )
