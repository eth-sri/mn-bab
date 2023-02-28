from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
import torch.nn as nn
from torch import Tensor

from src.abstract_domains.ai_util import AbstractElement
from src.abstract_layers.abstract_module import AbstractModule
from src.mn_bab_shape import AffineForm, MN_BaB_Shape
from src.utilities.config import BacksubstitutionConfig
from src.verification_subproblem import SubproblemState


class Flatten(nn.Flatten, AbstractModule):
    def __init__(
        self, start_dim: int, end_dim: int, input_dim: Tuple[int, ...]
    ) -> None:
        super(Flatten, self).__init__(start_dim, end_dim)  # type: ignore # mypy issue 4335
        self.input_dim = input_dim
        self.output_dim = (np.prod(input_dim),)

    @classmethod
    def from_concrete_module(  # type: ignore[override] # (checked at runtime)
        cls, module: nn.Flatten, input_dim: Tuple[int, ...], **kwargs: Any
    ) -> Flatten:
        assert isinstance(module, nn.Flatten)
        return cls(module.start_dim, module.end_dim, input_dim)

    def backsubstitute(
        self, config: BacksubstitutionConfig, abstract_shape: MN_BaB_Shape
    ) -> MN_BaB_Shape:

        # Dirty fix as storing an abstract network with input dim (784,) gives us a network with input dim (784,1)
        if len(self.input_dim) == 2 and self.input_dim[1] == 1:
            self.input_dim = (self.input_dim[0],)

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
        new_coef = affine_form.coef.view(*affine_form.coef.size()[:2], *self.input_dim)
        new_bias = affine_form.bias

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
        return self.forward(interval_lb), self.forward(interval_ub)

    def propagate_abstract_element(
        self,
        abs_input: AbstractElement,
        use_existing_bounds: Optional[bool] = None,
        activation_layer_only: bool = False,
        set_input: bool = True,
        set_output: bool = True,
    ) -> AbstractElement:
        return abs_input.flatten()
