from typing import Tuple

import torch
from torch import Tensor


class Permute(torch.nn.Module):
    def __init__(self, dims: Tuple[int, ...]) -> None:
        super(Permute, self).__init__()
        self.dims = dims

    def forward(self, x: Tensor) -> Tensor:
        return x.permute(self.dims)
