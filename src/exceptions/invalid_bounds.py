from typing import Sequence


class InvalidBoundsError(ValueError):
    """When intermediate bounds are set with lb > ub"""

    def __init__(self, invalid_bounds_mask_in_batch: Sequence[bool]):
        self.invalid_bounds_mask_in_batch = invalid_bounds_mask_in_batch
