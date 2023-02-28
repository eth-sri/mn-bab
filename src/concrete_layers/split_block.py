from typing import Optional, Tuple

import torch
from torch import Tensor
from torch import nn as nn


class SplitBlock(nn.Module):
    def __init__(
        self,
        split: Tuple[bool, Tuple[int, ...], Optional[int], int, bool],
        center_path: nn.Sequential,
        inner_reduce: Tuple[int, bool, bool],
        outer_reduce: Tuple[int, bool, bool],
    ) -> None:
        super(SplitBlock, self).__init__()
        self.center_path = center_path
        # (split.enable_pruning, split.split_size_or_sections, split.number_of_splits, split.dim, split.keep_size)
        self.split = split
        # (inner_reduce.dim, inner_reduce.keepdim, inner_reduce.noop_with_empty_axes)
        self.inner_reduce = inner_reduce
        self.outer_reduce = outer_reduce

    def forward(self, x: Tensor) -> Tensor:

        center_x, res_x = torch.split(
            x, split_size_or_sections=self.split[1], dim=self.split[3]
        )
        center_out = self.center_path(center_x)
        inner_merge = center_out * res_x

        inner_red = torch.sum(inner_merge, dim=self.inner_reduce[0])
        outer_red = torch.sum(res_x, dim=self.outer_reduce[0])

        out = inner_red / outer_red

        return out
