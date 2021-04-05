import numpy as np
from .tensor import Tensor


def tuplify(thing):
    try:
        return tuple(thing)
    except TypeError:
        return (thing,)


def one_hot(labels, num_classes):
    result = np.zeros([len(labels), num_classes])
    result[np.arange(len(labels)), labels] = 1
    return Tensor.from_numpy(result)
