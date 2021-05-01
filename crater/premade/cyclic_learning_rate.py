from dataclasses import dataclass


@dataclass(frozen=True)
class CyclicLearningRate:
    low: float
    high: float
    batches_per_cycle: int

    def learning_rate(self, batch_idx):
        offset = (batch_idx % self.batches_per_cycle) * 2
        if offset < self.batches_per_cycle:
            return self.low + (offset / self.batches_per_cycle) * (self.high - self.low)
        return self.high + (offset / self.batches_per_cycle - 1) * (
            self.low - self.high
        )
