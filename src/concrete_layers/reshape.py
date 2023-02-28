from typing import Tuple

import torch
from torch import Tensor


class Reshape(torch.nn.Module):
    def __init__(self, shape: Tuple[int, ...]) -> None:
        super(Reshape, self).__init__()
        # Assume that shape is without batch-size
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        return x.reshape((x.shape[0], *self.shape))
