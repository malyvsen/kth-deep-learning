from pathlib import Path
import pickle
import numpy as np
from PIL import Image
from crater import Tensor


def load_batch(filename: str):
    with (Path("../data") / filename).open("rb") as file:
        result = pickle.load(file, encoding="bytes")
    return dict(data=result[b"data"], labels=result[b"labels"])


def normalize_data(data: np.ndarray):
    return Tensor.from_numpy((data - data_mean) / data_std)


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


data = dict(
    train=load_batch("data_batch_1"),
    validation=load_batch("data_batch_2"),
    test=load_batch("test_batch"),
)


data_mean = np.mean(data["train"]["data"])
data_std = np.std(data["train"]["data"])
