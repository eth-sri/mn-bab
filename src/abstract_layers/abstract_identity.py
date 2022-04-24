from __future__ import annotations

from typing import Any, Tuple

import torch.nn as nn
from torch import Tensor

from src.abstract_layers.abstract_module import AbstractModule
from src.mn_bab_shape import MN_BaB_Shape


class Identity(nn.Identity, AbstractModule):
    def __init__(self, input_dim: Tuple[int, ...]) -> None:
        super(Identity, self).__init__()
        self.output_dim = input_dim
        self.dependence_set_block = False

    @classmethod
    def from_concrete_module(
        cls, module: nn.Identity, input_dim: Tuple[int, ...], **kwargs: Any
    ) -> Identity:
        return cls(input_dim)

    def backsubstitute(self, abstract_shape: MN_BaB_Shape) -> MN_BaB_Shape:
        return abstract_shape

    def propagate_interval(
        self, interval: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tensor]:
        return interval
