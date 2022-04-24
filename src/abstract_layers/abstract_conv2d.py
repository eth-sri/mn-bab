from __future__ import annotations

from math import floor
from typing import Any, Optional, Tuple, Union

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.abstract_layers.abstract_module import AbstractModule
from src.mn_bab_shape import MN_BaB_Shape
from src.utilities.dependence_sets import DependenceSets
from src.utilities.general import get_neg_pos_comp


class Conv2d(nn.Conv2d, AbstractModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, ...]],
        input_dim: Tuple[int, ...],
        stride: Union[int, Tuple[int, ...]] = 1,
        padding: Union[int, Tuple[int, ...]] = 0,
        dilation: Union[int, Tuple[int, ...]] = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super(Conv2d, self).__init__(  # type: ignore # mypy issue 4335
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        self.input_dim = input_dim
        output_height = floor(
            (
                input_dim[1]
                + 2 * self.padding[0]
                - self.dilation[0] * (self.kernel_size[0] - 1)
                - 1
            )
            / self.stride[0]
            + 1
        )
        output_width = floor(
            (
                input_dim[2]
                + 2 * self.padding[1]
                - self.dilation[1] * (self.kernel_size[1] - 1)
                - 1
            )
            / self.stride[1]
            + 1
        )
        self.output_dim = (out_channels, output_height, output_width)
        self.dependence_set_block = False

    @classmethod
    def from_concrete_module(
        cls, module: nn.Conv2d, input_dim: Tuple[int, ...], **kwargs: Any
    ) -> Conv2d:
        abstract_layer = cls(
            module.in_channels,
            module.out_channels,
            module.kernel_size,
            input_dim,
            module.stride,
            module.padding,
            module.dilation,
            module.groups,
            module.bias is not None,
        )
        abstract_layer.weight.data = module.weight.data
        if module.bias is not None:
            abstract_layer.bias.data = module.bias.data
        return abstract_layer

    def backsubstitute(self, abstract_shape: MN_BaB_Shape) -> MN_BaB_Shape:
        if abstract_shape.uses_dependence_sets():
            symmetric_stride = self.stride[0] == self.stride[1]
            symmetric_padding = self.padding[0] == self.padding[1]
            dilation_one = (
                self.dilation[0] == self.dilation[1] == 1 and self.dilation[0] == 1
            )
            group_one = self.groups == 1
            dependence_sets_assumptions = (
                symmetric_stride and symmetric_padding and dilation_one and group_one
            )
            assert dependence_sets_assumptions, "Dependence set assumptions violated."

            def backsubstitute_coef_and_bias(
                coef: DependenceSets, bias: Tensor
            ) -> Tuple[DependenceSets, Tensor]:
                new_bias = bias + (
                    0
                    if self.bias is None
                    else (coef.sets.sum((3, 4)) * self.bias).sum(2)
                )
                # [B*C*HW, c, d, d] -> [B*C*HW, c', d', d']
                new_coef_sets = F.conv_transpose2d(
                    coef.sets.flatten(end_dim=1), self.weight, stride=self.stride
                )
                new_coef = DependenceSets(
                    new_coef_sets.view(*coef.sets.shape[:2], *new_coef_sets.shape[1:]),
                    coef.spatial_idxs,
                    coef.cstride * self.stride[0],
                    coef.cpadding * self.stride[0] + self.padding[0],
                )
                return new_coef, new_bias

            new_lb_coef, new_lb_bias = backsubstitute_coef_and_bias(
                abstract_shape.lb_coef, abstract_shape.lb_bias
            )
            new_ub_coef, new_ub_bias = backsubstitute_coef_and_bias(
                abstract_shape.ub_coef, abstract_shape.ub_bias
            )
        else:
            """
            Adapted from PARC (https://gitlab.inf.ethz.ch/OU-VECHEV/PARC/-/blob/master/AIDomains/deeppoly.py)
            """
            assert isinstance(abstract_shape.lb_coef, Tensor)
            assert isinstance(abstract_shape.ub_coef, Tensor)

            kernel_wh = self.weight.shape[-2:]
            w_padding = (
                self.input_dim[1]
                + 2 * self.padding[0]
                - 1
                - self.dilation[0] * (kernel_wh[0] - 1)
            ) % self.stride[0]
            h_padding = (
                self.input_dim[2]
                + 2 * self.padding[1]
                - 1
                - self.dilation[1] * (kernel_wh[1] - 1)
            ) % self.stride[1]
            output_padding = (w_padding, h_padding)

            sz = abstract_shape.lb_coef.shape

            # process reference
            new_lb_bias = abstract_shape.lb_bias + (
                0
                if self.bias is None
                else (abstract_shape.lb_coef.sum((3, 4)) * self.bias).sum(2)
            )
            new_ub_bias = abstract_shape.ub_bias + (
                0
                if self.bias is None
                else (abstract_shape.ub_coef.sum((3, 4)) * self.bias).sum(2)
            )

            new_lb_coef = F.conv_transpose2d(
                abstract_shape.lb_coef.view((sz[0] * sz[1], *sz[2:])),
                self.weight,
                None,
                self.stride,
                self.padding,
                output_padding,
                self.groups,
                self.dilation,
            )
            new_ub_coef = F.conv_transpose2d(
                abstract_shape.ub_coef.view((sz[0] * sz[1], *sz[2:])),
                self.weight,
                None,
                self.stride,
                self.padding,
                output_padding,
                self.groups,
                self.dilation,
            )
            # F.pad(new_x_l_coef, (0, 0, w_padding, h_padding), "constant", 0)
            assert isinstance(new_lb_coef, Tensor)
            assert isinstance(new_ub_coef, Tensor)
            new_lb_coef = new_lb_coef.view((sz[0], sz[1], *new_lb_coef.shape[1:]))
            new_ub_coef = new_ub_coef.view((sz[0], sz[1], *new_ub_coef.shape[1:]))

        abstract_shape.update_bounds(new_lb_coef, new_ub_coef, new_lb_bias, new_ub_bias)
        return abstract_shape

    def propagate_interval(
        self, interval: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tensor]:
        interval_lb, interval_ub = interval

        neg_kernel, pos_kernel = get_neg_pos_comp(self.weight)

        def conv_with_kernel_and_bias(
            input: Tensor, kernel: Tensor, bias: Optional[Tensor]
        ) -> Tensor:
            return F.conv2d(
                input=input,
                weight=kernel,
                bias=bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )

        output_lb = conv_with_kernel_and_bias(
            interval_lb, pos_kernel, self.bias
        ) + conv_with_kernel_and_bias(interval_ub, neg_kernel, None)
        output_ub = conv_with_kernel_and_bias(
            interval_ub, pos_kernel, self.bias
        ) + conv_with_kernel_and_bias(interval_lb, neg_kernel, None)

        # assert (output_ub >= output_lb).all()

        return output_lb, output_ub
