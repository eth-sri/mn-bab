from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import torch.nn as nn
from torch import Tensor

from src.abstract_layers.abstract_module import AbstractModule
from src.mn_bab_shape import MN_BaB_Shape


class Flatten(nn.Flatten, AbstractModule):
    def __init__(
        self, start_dim: int, end_dim: int, input_dim: Tuple[int, ...]
    ) -> None:
        super(Flatten, self).__init__(start_dim, end_dim)  # type: ignore # mypy issue 4335
        self.input_dim = input_dim
        self.output_dim = np.prod(input_dim)

    @classmethod
    def from_concrete_module(
        cls, module: nn.Flatten, input_dim: Tuple[int, ...], **kwargs: Any
    ) -> Flatten:
        return cls(module.start_dim, module.end_dim, input_dim)

    def backsubstitute(self, abstract_shape: MN_BaB_Shape) -> MN_BaB_Shape:
        """
        Adapted from PARC (https://gitlab.inf.ethz.ch/OU-VECHEV/PARC/-/blob/master/AIDomains/deeppoly.py)
        """
        assert isinstance(abstract_shape.lb_coef, Tensor)
        assert isinstance(abstract_shape.ub_coef, Tensor)

        x_l_coef = abstract_shape.lb_coef.view(
            *abstract_shape.lb_coef.size()[:2], *self.input_dim
        )
        x_u_coef = abstract_shape.ub_coef.view(
            *abstract_shape.ub_coef.size()[:2], *self.input_dim
        )

        abstract_shape.update_bounds(
            x_l_coef,
            x_u_coef,
            abstract_shape.lb_bias,
            abstract_shape.ub_bias,
        )
        return abstract_shape

    def propagate_interval(
        self, interval: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tensor]:
        interval_lb, interval_ub = interval
        return self.forward(interval_lb), self.forward(interval_ub)
