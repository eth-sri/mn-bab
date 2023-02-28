from typing import Tuple

import torch
from torch import Tensor


class Concat(torch.nn.Module):
    def __init__(self, dim: int) -> None:
        super(Concat, self).__init__()
        self.dim = dim

    def forward(self, x: Tuple[Tensor, ...]) -> Tensor:
        return torch.cat(x, dim=self.dim)
