from __future__ import annotations

from math import floor
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.abstract_domains.ai_util import AbstractElement
from src.abstract_layers.abstract_module import AbstractModule
from src.mn_bab_shape import AffineForm, MN_BaB_Shape
from src.utilities.config import BacksubstitutionConfig
from src.utilities.dependence_sets import DependenceSets
from src.verification_subproblem import SubproblemState


class AvgPool2d(nn.AvgPool2d, AbstractModule):
    kernel_size: Tuple[int, int]  # type: ignore[assignment] # hack
    stride: Tuple[int, int]  # type: ignore[assignment]
    padding: Tuple[int, int]  # type: ignore[assignment]
    dilation: Tuple[int, int]  # type: ignore[assignment]

    weight: Tensor  # type: ignore[assignment]

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        input_dim: Tuple[int, ...],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
    ):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)

        super(AvgPool2d, self).__init__(  # type: ignore # mypy issue 4335
            kernel_size, stride, padding
        )

        self.input_dim = input_dim
        output_height = floor(
            (input_dim[1] + 2 * self.padding[0] - self.kernel_size[0]) / self.stride[0]
            + 1
        )
        output_width = floor(
            (input_dim[2] + 2 * self.padding[1] - self.kernel_size[1]) / self.stride[1]
            + 1
        )
        self.output_dim = (input_dim[0], output_height, output_width)
        self.dependence_set_block = False
        self.kernel_prod_norm = 1 / torch.prod(torch.Tensor(self.kernel_size))

    @classmethod
    def from_concrete_module(  # type: ignore[override] # checked at runtime
        cls, module: nn.AvgPool2d, input_dim: Tuple[int, ...], **kwargs: Any
    ) -> AvgPool2d:
        assert isinstance(module, nn.AvgPool2d)
        abstract_layer = cls(
            module.kernel_size, input_dim, module.stride, module.padding
        )
        return abstract_layer

    def backsubstitute(
        self, config: BacksubstitutionConfig, abstract_shape: MN_BaB_Shape
    ) -> MN_BaB_Shape:
        new_lb_form = self._backsub_affine_form(abstract_shape.lb, abstract_shape)
        new_ub_form: Optional[AffineForm] = None
        if abstract_shape.ub is not None:
            new_ub_form = self._backsub_affine_form(abstract_shape.ub, abstract_shape)

        abstract_shape.update_bounds(new_lb_form, new_ub_form)
        return abstract_shape

    def _backsub_affine_form(
        self, affine_form: AffineForm, abstract_shape: MN_BaB_Shape
    ) -> AffineForm:
        new_coef: Union[Tensor, DependenceSets]

        if abstract_shape.uses_dependence_sets():
            symmetric_stride = self.stride[0] == self.stride[1]
            symmetric_padding = self.padding[0] == self.padding[1]
            dilation_one = self.dilation[0] == self.dilation[1] == 1
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
                    coef.input_dim,
                    coef.cstride * self.stride[0],
                    coef.cpadding * self.stride[0] + self.padding[0],
                )
                return new_coef, new_bias

            assert isinstance(affine_form.coef, DependenceSets)
            new_coef, new_bias = backsubstitute_coef_and_bias(
                affine_form.coef, affine_form.bias
            )
        else:
            assert isinstance(affine_form.coef, Tensor)

            kernel_wh = self.kernel_size
            w_padding = (
                self.input_dim[1] + 2 * self.padding[0] - kernel_wh[0]
            ) % self.stride[0]
            h_padding = (
                self.input_dim[2] + 2 * self.padding[1] - kernel_wh[1]
            ) % self.stride[1]
            output_padding = (w_padding, h_padding)

            sz = affine_form.coef.shape
            weight = self.kernel_prod_norm * torch.ones(
                (self.input_dim[0], 1, *self.kernel_size), device=abstract_shape.device
            )

            new_bias = affine_form.bias

            new_coef = F.conv_transpose2d(
                affine_form.coef.view((sz[0] * sz[1], *sz[2:])),
                weight,
                None,
                self.stride,
                self.padding,
                output_padding,
                self.input_dim[0],
                1,
            )

            new_coef = new_coef.view((sz[0], sz[1], *new_coef.shape[1:]))  # type: ignore

        return AffineForm(new_coef, new_bias)

    def propagate_interval(
        self,
        interval: Tuple[Tensor, Tensor],
        use_existing_bounds: Optional[bool] = None,
        subproblem_state: Optional[SubproblemState] = None,
        activation_layer_only: bool = False,
        set_input: bool = True,
        set_output: bool = True,
    ) -> Tuple[Tensor, Tensor]:
        interval_lb, interval_ub = interval

        output_lb = F.avg_pool2d(
            interval_lb, self.kernel_size, self.stride, self.padding
        )
        output_ub = F.avg_pool2d(
            interval_ub, self.kernel_size, self.stride, self.padding
        )

        # assert (output_ub >= output_lb).all()

        return output_lb, output_ub

    def propagate_abstract_element(
        self,
        abs_input: AbstractElement,
        use_existing_bounds: Optional[bool] = None,
        activation_layer_only: bool = False,
        set_input: bool = True,
        set_output: bool = True,
    ) -> AbstractElement:
        assert all([self.kernel_size[0] == x for x in self.kernel_size])
        assert all([self.stride[0] == x for x in self.stride])
        assert all([self.padding[0] == x for x in self.padding])
        return abs_input.avg_pool2d(
            self.kernel_size[0], self.stride[0], self.padding[0]
        )
