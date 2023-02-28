from __future__ import annotations

from typing import Any, Optional, Tuple

# import torch
import torch.nn as nn
from torch import Tensor

from src.abstract_domains.ai_util import AbstractElement
from src.abstract_layers.abstract_module import AbstractModule
from src.mn_bab_shape import AffineForm, MN_BaB_Shape
from src.utilities.config import BacksubstitutionConfig
from src.utilities.general import get_neg_pos_comp
from src.verification_subproblem import SubproblemState


class Linear(nn.Linear, AbstractModule):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        input_dim: Tuple[int, ...],
    ) -> None:
        super(Linear, self).__init__(in_features, out_features, bias)  # type: ignore # mypy issue 4335
        self.output_dim = (*input_dim[:-1], out_features)

    @classmethod
    def from_concrete_module(  # type: ignore[override] # (checked at runtime)
        cls, module: nn.Linear, input_dim: Tuple[int, ...], **kwargs: Any
    ) -> Linear:
        assert isinstance(module, nn.Linear)
        abstract_module = cls(
            module.in_features, module.out_features, module.bias is not None, input_dim
        )
        abstract_module.weight.data = module.weight.data
        if module.bias is not None:
            abstract_module.bias.data = module.bias.data
        return abstract_module

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
        assert isinstance(affine_form.coef, Tensor)
        new_coef = affine_form.coef.matmul(self.weight)
        if self.bias is None:
            new_bias = 0
        else:
            new_bias = affine_form.coef.matmul(self.bias)
            if (
                len(new_bias.shape) == 3
            ):  # in case we have a matmul on the last dimension the bias is otherwise over multiple channels
                new_bias = new_bias.sum(dim=2)
        new_bias += affine_form.bias
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

    def propagate_abstract_element(
        self,
        abs_input: AbstractElement,
        use_existing_bounds: Optional[bool] = None,
        activation_layer_only: bool = False,
        set_input: bool = True,
        set_output: bool = True,
    ) -> AbstractElement:
        return abs_input.linear(self.weight, self.bias)
