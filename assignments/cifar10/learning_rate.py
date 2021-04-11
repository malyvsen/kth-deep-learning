def cyclic(*, low, high, half_cycle_size):
    def result(num_examples_seen):
        offset = num_examples_seen % half_cycle_size
        if offset % 2 == 0:
            return low + (offset / half_cycle_size) * (high - low)
        return high + (offset / half_cycle_size) * (low - high)

    return result
