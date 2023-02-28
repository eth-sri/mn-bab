from __future__ import annotations

from typing import Any, Optional, Tuple

import torch.nn.functional as F
from torch import Tensor

from src.abstract_domains.ai_util import AbstractElement
from src.abstract_layers.abstract_module import AbstractModule
from src.concrete_layers.pad import Pad as concretePad
from src.mn_bab_shape import AffineForm, MN_BaB_Shape
from src.utilities.config import BacksubstitutionConfig
from src.verification_subproblem import SubproblemState


class Pad(concretePad, AbstractModule):
    def __init__(
        self,
        pad: Tuple[int, ...],
        input_dim: Tuple[int, ...],
        mode: str = "constant",
        value: float = 0.0,
    ):

        if isinstance(pad, int):
            pad = (pad, pad, pad, pad)
        elif len(pad) == 2:
            pad = (pad[0], pad[1], 0, 0)

        super(Pad, self).__init__(pad, mode, value)
        self.input_dim = input_dim
        self.output_dim = (
            input_dim[0],
            input_dim[1] + pad[2] + pad[3],
            input_dim[2] + pad[0] + pad[1],
        )

    @classmethod
    def from_concrete_module(  # type: ignore[override] # (checked at runtime)
        cls, module: concretePad, input_dim: Tuple[int, ...], **kwargs: Any
    ) -> Pad:
        assert isinstance(module, concretePad)
        abstract_layer = cls(
            module.pad,
            input_dim,
            module.mode,
            module.value,
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
        if abstract_shape.uses_dependence_sets():
            assert False, "Not implemented - Pad with dependence sets"
        assert isinstance(affine_form.coef, Tensor)
        assert len(self.pad) == 4, f"Incompatible padding size: {self.pad}"
        pad_l, pad_r, pad_t, pad_b = self.pad
        pad_b = affine_form.coef.shape[3] - pad_b
        pad_r = affine_form.coef.shape[4] - pad_r

        # Step 1 unpad the coefficients as the outer coeffs refer to padding values
        new_coef = affine_form.coef[:, :, :, pad_t:pad_b, pad_l:pad_r]
        # Step 2 concretize the contribution of the padding into the bias
        only_pad_coef = (
            affine_form.coef.detach().clone()
        )  # TODO @Robin Detach required?
        only_pad_coef[:, :, :, pad_t:pad_b, pad_l:pad_r] = 0
        only_pad_coef *= self.value
        only_pad_contr = only_pad_coef.sum((2, 3, 4))
        new_bias = affine_form.bias + only_pad_contr

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
        output_lb = F.pad(interval_lb, self.pad, self.mode, self.value)
        output_ub = F.pad(interval_ub, self.pad, self.mode, self.value)

        return output_lb, output_ub

    def propagate_abstract_element(
        self,
        abs_input: AbstractElement,
        use_existing_bounds: Optional[bool] = None,
        activation_layer_only: bool = False,
        set_input: bool = True,
        set_output: bool = True,
    ) -> AbstractElement:
        return abs_input.pad(self.pad, self.mode, self.value)
