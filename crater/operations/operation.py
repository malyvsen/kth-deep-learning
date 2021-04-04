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
        operation.output = operation.forward(*args, **kwargs)
        for recorder in active_recorders:
            recorder.register(operation)
        return operation.output

    def gradients(self, outgoing_gradient: np.ndarray):
        assert outgoing_gradient.shape == self.output.shape
        return self.backward(outgoing_gradient)
