from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


class Pad(torch.nn.Module):
    def __init__(
        self, pad: Tuple[int, ...], mode: str = "constant", value: float = 0.0
    ) -> None:
        super(Pad, self).__init__()
        self.pad = pad if pad is not None else (0, 0, 0, 0)
        self.mode = mode
        self.value = value

    def forward(self, x: Tensor) -> Tensor:
        return F.pad(x, self.pad, self.mode, self.value)
