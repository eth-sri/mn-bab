from torch import Tensor
from torch import nn as nn


class ResidualBlock(nn.Module):
    def __init__(
        self,
        path_a: nn.Sequential,
        path_b: nn.Sequential,
    ) -> None:
        super(ResidualBlock, self).__init__()
        self.path_a = path_a
        self.path_b = path_b

    def forward(self, x: Tensor) -> Tensor:
        out = self.path_a(x) + self.path_b(x)
        return out
