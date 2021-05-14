from typing import List
from dataclasses import dataclass
from functools import cached_property
import pandas as pd
from .tweet import Tweet


@dataclass(frozen=True)
class Tweets:
    texts: List[str]

    @classmethod
    def from_csv(cls, path):
        dataframe = pd.read_csv(path)
        return cls(texts=list(dataframe["text"]))

    def __iter__(self):
        for text in self.texts:
            yield Tweet.full(character_ids=[self.character_ids[char] for char in text])

    @cached_property
    def num_unique_characters(self):
        return len(self.unique_characters)

    @cached_property
    def character_ids(self):
        return {character: idx for idx, character in enumerate(self.unique_characters)}

    @cached_property
    def id_characters(self):
        return {idx: character for character, idx in self.character_ids.items()}

    @cached_property
    def unique_characters(self):
        return set("".join(self.texts))
