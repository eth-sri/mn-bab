from __future__ import annotations

from typing import Any, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from src.abstract_layers.abstract_module import AbstractModule
from src.mn_bab_shape import MN_BaB_Shape
from src.utilities.dependence_sets import DependenceSets
from src.utilities.general import get_neg_pos_comp


class BatchNorm2d(nn.BatchNorm2d, AbstractModule):
    def __init__(
        self,
        in_channels: int,
        input_dim: Tuple[int, ...],
        affine: bool = True,
    ):
        super(BatchNorm2d, self).__init__(in_channels, affine=affine)  # type: ignore
        self.input_dim = input_dim
        self.output_dim = input_dim
        self.dependence_set_block = False

    @classmethod
    def from_concrete_module(
        cls, module: nn.BatchNorm2d, input_dim: Tuple[int, ...], **kwargs: Any
    ) -> BatchNorm2d:
        abstract_layer = cls(
            module.num_features,
            input_dim,
            module.affine,
        )
        abstract_layer.running_var.data = module.running_var.data
        abstract_layer.running_mean.data = module.running_mean.data
        if module.affine:
            abstract_layer.weight.data = module.weight.data
            abstract_layer.bias.data = module.bias.data

        abstract_layer.track_running_stats = module.track_running_stats
        abstract_layer.training = False

        D = module.eps

        abstract_layer.mult_term = (
            (
                (abstract_layer.weight if abstract_layer.affine else 1)
                / torch.sqrt(abstract_layer.running_var + D)
            )
            .detach()
            .requires_grad_(False)
        )
        abstract_layer.add_term = (
            (
                (abstract_layer.bias if abstract_layer.affine else 0)
                - abstract_layer.running_mean * abstract_layer.mult_term
            )
            .detach()
            .requires_grad_(False)
        )

        return abstract_layer

    def backsubstitute(self, abstract_shape: MN_BaB_Shape) -> MN_BaB_Shape:
        if abstract_shape.uses_dependence_sets():

            def backsubstitute_coef_and_bias(
                coef: DependenceSets, bias: Tensor
            ) -> Tuple[DependenceSets, Tensor]:
                new_bias = bias + ((coef.sets.sum((3, 4)) * self.add_term).sum(2))
                # [B,C,HW, c, d, d] -> [B*C*HW, c', d', d']
                new_coef_sets = coef.sets.flatten(end_dim=1) * self.mult_term.unsqueeze(
                    0
                ).unsqueeze(-1).unsqueeze(-1)

                new_coef = DependenceSets(
                    new_coef_sets.view(*coef.sets.shape[:2], *new_coef_sets.shape[1:]),
                    coef.spatial_idxs,
                    coef.cstride,
                    coef.cpadding,
                )
                return new_coef, new_bias

            new_lb_coef, new_lb_bias = backsubstitute_coef_and_bias(
                abstract_shape.lb_coef, abstract_shape.lb_bias
            )
            new_ub_coef, new_ub_bias = backsubstitute_coef_and_bias(
                abstract_shape.ub_coef, abstract_shape.ub_bias
            )
        else:
            assert isinstance(abstract_shape.lb_coef, Tensor)
            assert isinstance(abstract_shape.ub_coef, Tensor)

            D = self.eps
            mult_term = (
                ((self.weight if self.affine else 1) / torch.sqrt(self.running_var + D))
                .detach()
                .requires_grad_(False)
            )
            add_term = (
                ((self.bias if self.affine else 0) - self.running_mean * self.mult_term)
                .detach()
                .requires_grad_(False)
            )

            assert (self.mult_term == mult_term).all()
            assert (self.add_term == add_term).all()

            # process reference
            new_lb_bias = abstract_shape.lb_bias + (
                (abstract_shape.lb_coef.sum((3, 4)) * self.add_term).sum(2)
            )
            new_ub_bias = abstract_shape.ub_bias + (
                (abstract_shape.ub_coef.sum((3, 4)) * self.add_term).sum(2)
            )

            new_lb_coef = abstract_shape.lb_coef * self.mult_term.view(1, 1, -1, 1, 1)
            new_ub_coef = abstract_shape.ub_coef * self.mult_term.view(1, 1, -1, 1, 1)
            assert isinstance(new_lb_coef, Tensor)
            assert isinstance(new_ub_coef, Tensor)

        abstract_shape.update_bounds(new_lb_coef, new_ub_coef, new_lb_bias, new_ub_bias)
        return abstract_shape

    def propagate_interval(
        self, interval: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tensor]:
        interval_lb, interval_ub = interval

        neg_kernel, pos_kernel = get_neg_pos_comp(self.mult_term.view(1, -1, 1, 1))

        output_lb = (
            interval_lb * pos_kernel
            + interval_ub * neg_kernel
            + self.add_term.view(1, -1, 1, 1)
        )
        output_ub = (
            interval_lb * neg_kernel
            + interval_ub * pos_kernel
            + self.add_term.view(1, -1, 1, 1)
        )

        # D = self.eps
        # mult_term = ((self.weight if self.affine else 1) / torch.sqrt(self.running_var + D)).detach().requires_grad_(False)
        # add_term = ((self.bias if self.affine else 0) - self.running_mean * self.mult_term).detach().requires_grad_(False)
        #
        # assert ((self.mult_term-mult_term)==0).all()
        # assert ((self.add_term - add_term) == 0).all()
        #
        # assert (output_ub >= output_lb).all()

        return output_lb, output_ub
