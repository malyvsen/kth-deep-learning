from crater.tensor import Tensor


def softmax(tensor: Tensor, axes=None):
    exp_tensor = tensor.exp()
    return exp_tensor / exp_tensor.sum(axes=axes).expand_dims(axes=axes)


Tensor.softmax = softmax
