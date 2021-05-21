import numpy as np


def one_hot(labels: np.ndarray, num_classes: int):
    assert len(np.shape(labels)) == 1
    result = np.zeros([len(labels), num_classes])
    result[np.arange(len(labels)), labels] = 1
    return result


def matmul_backward(gradient: np.ndarray, left: np.ndarray, right: np.ndarray):
    return (
        np.matmul(gradient, right.T),  # gradient wrt left operand
        np.matmul(left.T, gradient),  # gradient wrt right operand
    )


def tanh(input: np.ndarray):
    return np.tanh(input)


def tanh_backward(gradient: np.ndarray, input: np.ndarray):
    return gradient / np.cosh(input) ** 2


def softmax(input: np.ndarray, axis: int):
    exp_input = np.exp(input)
    return exp_input / exp_input.sum(axis=axis, keepdims=True)


def softmax_backward(gradient: np.ndarray, input: np.ndarray, axis: int):
    output = softmax(input, axis=axis)
    return output * (gradient - np.sum(gradient * output, axis=axis))
