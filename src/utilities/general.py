from typing import Sequence, Tuple

import torch
from torch import Tensor


def all_larger_equal(seq: Sequence, threshold: float) -> bool:
    return all(el >= threshold for el in seq)


def any_smaller(seq: Sequence, threshold: float) -> bool:
    return any(el < threshold for el in seq)


def get_neg_pos_comp(x: Tensor) -> Tuple[Tensor, Tensor]:
    neg_comp = torch.where(x < 0, x, torch.zeros_like(x))
    pos_comp = torch.where(x >= 0, x, torch.zeros_like(x))
    return neg_comp, pos_comp
