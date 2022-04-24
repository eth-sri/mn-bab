from __future__ import annotations

from typing import Any, Sequence, Tuple

import torch
from torch import Tensor

import src.concrete_layers.normalize as concrete_normalize
from src.abstract_layers.abstract_module import AbstractModule
from src.mn_bab_shape import MN_BaB_Shape


class Normalization(concrete_normalize.Normalize, AbstractModule):
    def __init__(
        self,
        means: Sequence[float],
        stds: Sequence[float],
        device: torch.device,
        channel_dim: int,
        output_dim: Tuple[int, ...],
    ) -> None:
        super(Normalization, self).__init__(means, stds, channel_dim)  # type: ignore # mypy issue 4335
        super(Normalization, self).to(device)
        self.output_dim = output_dim
        self.dependence_set_block = False

    @classmethod
    def from_concrete_module(
        cls,
        module: concrete_normalize.Normalize,
        input_dim: Tuple[int, ...],
        **kwargs: Any
    ) -> Normalization:
        return cls(
            module.means.flatten().tolist(),
            module.stds.flatten().tolist(),
            module.means.device,
            module.channel_dim,
            input_dim,
        )

    def backsubstitute(self, abstract_shape: MN_BaB_Shape) -> MN_BaB_Shape:
        """
        Adapted from PARC (https://gitlab.inf.ethz.ch/OU-VECHEV/PARC/-/blob/master/AIDomains/deeppoly.py)
        """
        assert isinstance(abstract_shape.lb_coef, Tensor)
        assert isinstance(abstract_shape.ub_coef, Tensor)

        req_shape = [1] * abstract_shape.lb_coef.dim()
        req_shape[2] = self.means.numel()

        x_l_bias = abstract_shape.lb_bias + (
            abstract_shape.lb_coef * (-self.means / self.stds).view(req_shape)
        ).view(*abstract_shape.lb_coef.size()[:2], -1).sum(2)
        x_u_bias = abstract_shape.ub_bias + (
            abstract_shape.ub_coef * (-self.means / self.stds).view(req_shape)
        ).view(*abstract_shape.ub_coef.size()[:2], -1).sum(2)

        x_l_coef = abstract_shape.lb_coef / self.stds.view(req_shape)
        x_u_coef = abstract_shape.ub_coef / self.stds.view(req_shape)

        abstract_shape.update_bounds(x_l_coef, x_u_coef, x_l_bias, x_u_bias)
        return abstract_shape

    def propagate_interval(
        self, interval: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tensor]:
        interval_lb, interval_ub = interval

        output_lb, output_ub = self.forward(interval_lb), self.forward(interval_ub)
        # assert (output_ub >= output_lb).all()

        return output_lb, output_ub
