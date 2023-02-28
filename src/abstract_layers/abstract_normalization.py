from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import torch
from torch import Tensor

import src.concrete_layers.normalize as concrete_normalize
from src.abstract_domains.ai_util import AbstractElement
from src.abstract_layers.abstract_module import AbstractModule
from src.mn_bab_shape import AffineForm, MN_BaB_Shape
from src.utilities.config import BacksubstitutionConfig
from src.verification_subproblem import SubproblemState


class Normalization(concrete_normalize.Normalize, AbstractModule):
    mean: Tensor
    sigma: Tensor
    channel_dim: int
    output_dim: Tuple[int, ...]
    dependence_set_block: bool

    def __init__(
        self,
        means: Sequence[float],
        stds: Sequence[float],
        device: torch.device,
        channel_dim: int,
        output_dim: Tuple[int, ...],
    ) -> None:
        super(Normalization, self).__init__(means, stds, channel_dim)
        super(Normalization, self).to(device)
        self.output_dim = output_dim
        self.dependence_set_block = False

    @classmethod
    def from_concrete_module(  # type: ignore[override] # (checked at runtime)
        cls,
        module: concrete_normalize.Normalize,
        input_dim: Tuple[int, ...],
        **kwargs: Any,
    ) -> Normalization:
        assert isinstance(module, concrete_normalize.Normalize)
        return cls(
            module.means.flatten().tolist(),
            module.stds.flatten().tolist(),
            module.means.device,
            module.channel_dim,
            input_dim,
        )

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
        req_shape = [1] * affine_form.coef.dim()
        req_shape[2] = self.means.numel()

        new_bias = affine_form.bias + (
            affine_form.coef * (-self.means / self.stds).view(req_shape)
        ).view(*affine_form.coef.size()[:2], -1).sum(2)

        new_coef = affine_form.coef / self.stds.view(req_shape)

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

        output_lb, output_ub = self.forward(interval_lb), self.forward(interval_ub)
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
        return abs_input.normalize(self.means, self.stds)
