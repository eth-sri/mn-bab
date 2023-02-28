from __future__ import annotations

import typing
from typing import Any, Callable, Optional, Tuple, Union

import torch
from torch import Tensor

from src.abstract_layers.abstract_module import AbstractModule
from src.concrete_layers import unbinary_op
from src.mn_bab_shape import AffineForm, MN_BaB_Shape
from src.utilities.config import BacksubstitutionConfig
from src.utilities.dependence_sets import DependenceSets
from src.utilities.general import tensor_reduce
from src.verification_subproblem import SubproblemState

EPS = 1e-15


class UnbinaryOp(AbstractModule):
    def __init__(
        self, op: str, const_val: Tensor, apply_right: bool, input_dim: Tuple[int, ...]
    ) -> None:
        super(UnbinaryOp, self).__init__()
        self.op = op
        # self.const_val = torch.nn.Parameter(const_val,requires_grad=False)
        self.register_buffer(
            "const_val",
            const_val,
            persistent=False,
        )
        self.apply_right = apply_right
        self.output_dim = input_dim

    @classmethod
    @typing.no_type_check  # Mypy can't handle the buffer type
    def from_concrete_module(
        cls, module: unbinary_op.UnbinaryOp, input_dim: Tuple[int, ...], **kwargs: Any
    ) -> UnbinaryOp:
        return cls(module.op, module.const_val, module.apply_right, input_dim)

    @typing.no_type_check  # Mypy can't handle the buffer type
    def backsubstitute(
        self,
        config: BacksubstitutionConfig,
        abstract_shape: MN_BaB_Shape,
        intermediate_bounds_callback: Optional[
            Callable[[Tensor], Tuple[Tensor, Tensor]]
        ] = None,
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
        if abstract_shape.uses_dependence_sets():
            assert isinstance(affine_form.coef, DependenceSets)
            coef = affine_form.coef.sets
        else:
            assert isinstance(affine_form.coef, Tensor)
            coef = affine_form.coef

        bias = affine_form.bias

        bias_c = (coef * self.const_val).view((*coef.shape[:2], -1)).sum(dim=2)
        if self.op == "add":
            bias = bias + bias_c
        elif self.op == "sub":
            if self.apply_right:  # Y = C - X
                coef *= -1
                bias = bias + bias_c
            else:  # X - C
                bias = bias - bias_c
        elif self.op == "mul":
            coef *= self.const_val
        elif self.op == "div":
            if self.apply_right:
                assert False, "Tried to apply non linear division operation"
            else:
                coef /= self.const_val

        final_coef: Union[Tensor, DependenceSets] = coef
        if abstract_shape.uses_dependence_sets():
            assert isinstance(affine_form.coef, DependenceSets)
            final_coef = DependenceSets(
                coef,
                affine_form.coef.spatial_idxs,
                affine_form.coef.input_dim,
                affine_form.coef.cstride,
                affine_form.coef.cpadding,
            )

        return AffineForm(final_coef, bias)

    @typing.no_type_check  # Mypy can't handle the buffer type
    def forward(self, x: Tensor):
        const_val = self.const_val  # .squeeze()
        if self.apply_right:
            left, right = const_val, x
        else:
            left, right = x, const_val

        if self.op == "add":
            return left + right
        elif self.op == "sub":
            return left - right
        elif self.op == "mul":
            return left * right
        elif self.op == "div":
            return left / right
        else:
            assert False, f"Unknown operation {self.op}"

    @typing.no_type_check  # Mypy can't handle the buffer type
    def propagate_interval(
        self,
        interval: Tuple[Tensor, Tensor],
        use_existing_bounds: Optional[bool] = None,
        subproblem_state: Optional[SubproblemState] = None,
        activation_layer_only: bool = False,
        set_input: bool = True,
        set_output: bool = True,
    ) -> Tuple[Tensor, Tensor]:

        lb, ub = interval

        if self.apply_right:
            left_lb, right_lb = self.const_val, lb
            left_ub, right_ub = self.const_val, ub
        else:
            left_lb, right_lb = lb, self.const_val
            left_ub, right_ub = ub, self.const_val

        if self.op == "add":
            return (left_lb + right_lb, left_ub + right_ub)
        elif self.op == "sub":
            return (left_lb - right_ub, left_ub - right_lb)
        elif self.op == "mul":
            ll = left_lb * right_lb
            lu = left_lb * right_ub
            ul = left_ub * right_lb
            uu = left_ub * right_ub
            return (
                tensor_reduce(torch.minimum, [ll, lu, ul, uu]),
                tensor_reduce(torch.maximum, [ll, lu, ul, uu]),
            )
        elif self.op == "div":
            assert self.const_val != 0, "No division by 0"
            ll = left_lb / right_lb
            lu = left_lb / right_ub
            ul = left_ub / right_lb
            uu = left_ub / right_ub
            left = tensor_reduce(torch.minimum, [ll, lu, ul, uu])
            right = tensor_reduce(torch.maximum, [ll, lu, ul, uu])
            return (left, right)
        else:
            assert False, f"Unknown operation {self.op}"
