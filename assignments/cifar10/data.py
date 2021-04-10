from pathlib import Path
import pickle
import numpy as np
from PIL import Image
from crater import Tensor


def load_batch(filename: str):
    with (Path("../data") / filename).open("rb") as file:
        result = pickle.load(file, encoding="bytes")
    return dict(features=result[b"data"], labels=result[b"labels"])


def make_normalizer(train_features: np.ndarray):
    mean = np.mean(train_features)
    std = np.std(train_features)
    return lambda features: Tensor.from_numpy((features - mean) / std)


def vector_to_image(*vectors: np.ndarray):
    concatenated = np.concatenate(
        [
            thing
            for vector in vectors
            for thing in [
                np.reshape(
                    (vector - vector.min()) / (vector.max() - vector.min()),
                    [3, 32, 32],
                ),
                np.ones([3, 32, 4]),
            ]
        ],
        axis=2,
    )
    return Image.fromarray(
        (np.moveaxis(concatenated, 0, -1) * 255).astype(np.uint8)
    ).resize([dim * 2 for dim in concatenated.shape[1:][::-1]], resample=Image.NEAREST)
