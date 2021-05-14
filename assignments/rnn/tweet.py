from typing import List
from dataclasses import dataclass


@dataclass(frozen=True)
class Tweet:
    start_count: int
    character_ids: List[int]

    @classmethod
    def full(cls, character_ids: List[int]) -> "Tweet":
        return cls(start_count=0, character_ids=character_ids)

    def fragments(self, length: int):
        for start in range(0, len(self.character_ids) - 1, length):
            yield type(self)(
                start_count=start,
                character_ids=self.character_ids[start : start + length + 1],
            )

    @property
    def context(self):
        return self.character_ids[:-1]

    @property
    def char_counts(self):
        return range(self.start_count, self.start_count + len(self.character_ids) - 1)

    @property
    def targets(self):
        return self.character_ids[1:]
