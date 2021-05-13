from typing import List
from dataclasses import dataclass


@dataclass(frozen=True)
class Tweet:
    character_ids: List[int]

    @property
    def context(self):
        return self.character_ids[:-1]

    @property
    def ttls(self):
        return range(len(self.character_ids) - 1, 0, -1)

    @property
    def targets(self):
        return self.character_ids[1:]
