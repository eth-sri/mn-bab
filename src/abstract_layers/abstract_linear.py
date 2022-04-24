from __future__ import annotations

from typing import Any, Tuple

import torch.nn as nn
from torch import Tensor

from src.abstract_layers.abstract_module import AbstractModule
from src.mn_bab_shape import MN_BaB_Shape
from src.utilities.general import get_neg_pos_comp


class Linear(nn.Linear, AbstractModule):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(Linear, self).__init__(in_features, out_features, bias)  # type: ignore # mypy issue 4335
        self.output_dim = (out_features,)

    @classmethod
    def from_concrete_module(
        cls, module: nn.Linear, input_dim: Tuple[int, ...], **kwargs: Any
    ) -> Linear:
        abstract_module = cls(
            module.in_features, module.out_features, module.bias is not None
        )
        abstract_module.weight.data = module.weight.data
        abstract_module.bias.data = module.bias.data
        return abstract_module

    def backsubstitute(self, abstract_shape: MN_BaB_Shape) -> MN_BaB_Shape:
        assert isinstance(abstract_shape.lb_coef, Tensor)
        assert isinstance(abstract_shape.ub_coef, Tensor)

        new_lb_coef = abstract_shape.lb_coef.matmul(self.weight)
        new_ub_coef = abstract_shape.ub_coef.matmul(self.weight)

        new_lb_bias = (
            0 if self.bias is None else abstract_shape.lb_coef.matmul(self.bias)
        ) + abstract_shape.lb_bias
        new_ub_bias = (
            0 if self.bias is None else abstract_shape.ub_coef.matmul(self.bias)
        ) + abstract_shape.ub_bias

        abstract_shape.update_bounds(new_lb_coef, new_ub_coef, new_lb_bias, new_ub_bias)
        return abstract_shape

    def propagate_interval(
        self, interval: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tensor]:
        interval_lb, interval_ub = interval

        neg_weight, pos_weight = get_neg_pos_comp(self.weight.unsqueeze(0))

        output_lb = (
            pos_weight.matmul(interval_lb.unsqueeze(-1))
            + neg_weight.matmul(interval_ub.unsqueeze(-1))
        ).squeeze(dim=-1) + self.bias
        output_ub = (
            pos_weight.matmul(interval_ub.unsqueeze(-1))
            + neg_weight.matmul(interval_lb.unsqueeze(-1))
        ).squeeze(dim=-1) + self.bias

        # assert (output_ub >= output_lb).all()

        return output_lb, output_ub
