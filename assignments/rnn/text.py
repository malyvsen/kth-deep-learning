from dataclasses import dataclass
from functools import cached_property
import numpy as np
from .passage import Passage


@dataclass(frozen=True)
class Text:
    text: str

    @classmethod
    def from_file(cls, path):
        with open(path, "r", encoding="utf-8") as file:
            return cls(text=file.read())

    def passages(self, length: int):
        for idx in range(length, len(self.text) - 1):
            yield Passage(
                context=self.text_ids[idx - length : idx],
                targets=self.text_ids[idx - length + 1 : idx + 1],
            )

    @cached_property
    def num_unique_characters(self):
        return len(set(self.text))

    @cached_property
    def text_ids(self):
        return np.array([self.character_ids[char] for char in self.text], dtype=int)

    @cached_property
    def character_ids(self):
        return {character: idx for idx, character in enumerate(set(self.text))}

    @cached_property
    def id_characters(self):
        return {idx: character for character, idx in self.character_ids.items()}
