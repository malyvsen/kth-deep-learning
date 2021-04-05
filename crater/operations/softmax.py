def softmax(tensor):
    exp_tensor = tensor.exp()
    return exp_tensor / exp_tensor.sum()
