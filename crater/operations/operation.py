from inspect import signature
import numpy as np
from crater.recorder import active_recorders


class Operation:
    def __init__(self):
        pass

    @classmethod
    def apply(cls, *args, **kwargs):
        operation = cls()
        bound_args = signature(cls.forward).bind(operation, *args, **kwargs)
        bound_args.apply_defaults()
        for argument, value in bound_args.arguments.items():
            setattr(operation, argument, value)
        output = operation.forward(*args, **kwargs)
        for recorder in active_recorders:
            recorder.register(operation=operation, output=output)
        return output
