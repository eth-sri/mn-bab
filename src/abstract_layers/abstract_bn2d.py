from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from src.abstract_domains.ai_util import AbstractElement
from src.abstract_layers.abstract_module import AbstractModule
from src.mn_bab_shape import AffineForm, MN_BaB_Shape
from src.utilities.config import BacksubstitutionConfig
from src.utilities.dependence_sets import DependenceSets
from src.utilities.general import get_neg_pos_comp
from src.verification_subproblem import SubproblemState


class BatchNorm2d(nn.BatchNorm2d, AbstractModule):
    mult_term: Tensor
    add_term: Tensor
    weight: nn.Parameter
    running_var: Tensor
    running_mean: Tensor
    current_mean: Tensor
    current_var: Tensor

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
    def from_concrete_module(  # type: ignore[override] # (checked at runtime)
        cls, module: nn.BatchNorm2d, input_dim: Tuple[int, ...], **kwargs: Any
    ) -> BatchNorm2d:
        assert isinstance(module, nn.BatchNorm2d)
        abstract_layer = cls(
            module.num_features,
            input_dim,
            module.affine,
        )
        assert abstract_layer.running_var is not None
        assert module.running_var is not None
        abstract_layer.running_var.data = module.running_var.data
        assert abstract_layer.running_mean is not None
        assert module.running_mean is not None
        abstract_layer.running_mean.data = module.running_mean.data
        if module.affine:
            abstract_layer.weight.data = module.weight.data
            abstract_layer.bias.data = module.bias.data

        abstract_layer.track_running_stats = module.track_running_stats
        abstract_layer.training = False

        D = module.eps

        mult_term = (
            (
                (abstract_layer.weight if abstract_layer.affine else 1)
                / torch.sqrt(abstract_layer.running_var + D)
            )
            .detach()
            .requires_grad_(False)
        )
        abstract_layer.register_buffer("mult_term", mult_term)
        add_term = (
            (
                (abstract_layer.bias if abstract_layer.affine else 0)
                - abstract_layer.running_mean * abstract_layer.mult_term
            )
            .detach()
            .requires_grad_(False)
        )
        abstract_layer.register_buffer("add_term", add_term)

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
        if isinstance(affine_form.coef, Tensor):
            new_bias = affine_form.bias + (
                (affine_form.coef.sum((3, 4)) * self.add_term).sum(2)
            )
            new_coef = affine_form.coef * self.mult_term.view(1, 1, -1, 1, 1)
        elif isinstance(affine_form.coef, DependenceSets):
            new_bias = affine_form.bias + (
                DependenceSets.unfold_to(
                    self.add_term.unsqueeze(0)
                    .unsqueeze(2)
                    .unsqueeze(3)
                    .expand(affine_form.bias.shape[0], *self.input_dim),
                    affine_form.coef,
                )
                * affine_form.coef.sets
            ).sum((2, 3, 4))
            new_coef = affine_form.coef.sets * self.mult_term.view(1, 1, -1, 1, 1)
            new_coef = DependenceSets(
                new_coef,
                affine_form.coef.spatial_idxs,
                affine_form.coef.input_dim,
                affine_form.coef.cstride,
                affine_form.coef.cpadding,
            )
        else:
            assert False, "AffineForm not recognized"

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

    def propagate_abstract_element(
        self,
        abs_input: AbstractElement,
        use_existing_bounds: Optional[bool] = None,
        activation_layer_only: bool = False,
        set_input: bool = True,
        set_output: bool = True,
    ) -> AbstractElement:
        return abs_input.batch_norm(self)
