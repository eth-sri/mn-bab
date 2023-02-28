from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union

import torch
from torch import Tensor

from src.abstract_domains.ai_util import AbstractElement
from src.abstract_layers.abstract_module import AbstractModule
from src.concrete_layers.slice import Slice as concreteSlice
from src.mn_bab_shape import AffineForm, MN_BaB_Shape
from src.state.subproblem_state import SubproblemState
from src.utilities.config import BacksubstitutionConfig


class Slice(concreteSlice, AbstractModule):
    def __init__(
        self,
        starts: int,
        ends: int,
        dim: int,
        steps: int,
        input_dim: Tuple[int, ...],
    ):
        super(Slice, self).__init__(dim, starts, ends, steps)
        self.starts = starts
        self.ends = ends
        self.abs_dim = dim - 1  # This dim does not include batch-size
        self.steps = steps

        # The only allowed neg. end is -1 which signals that we go till the end
        if self.ends < 0:
            assert self.ends == -1, "Negative slice ending != -1"
            self.ends = input_dim[self.abs_dim]

        self.input_dim = input_dim
        output_dim = list(input_dim)
        output_dim[self.abs_dim] = len(range(self.starts, self.ends, steps))
        self.output_dim = tuple(output_dim)

    @classmethod
    def from_concrete_module(  # type: ignore[override] # (checked at runtime)
        cls, module: concreteSlice, input_dim: Tuple[int, ...], **kwargs: Any
    ) -> Slice:
        assert isinstance(module, concreteSlice)
        abstract_layer = cls(
            module.starts, module.ends, module.dim, module.steps, input_dim
        )
        return abstract_layer

    def backsubstitute(  # type: ignore[override]
        self,
        config: BacksubstitutionConfig,
        abstract_shape: Union[MN_BaB_Shape, List[MN_BaB_Shape]],
    ) -> MN_BaB_Shape:

        new_ub_form: Optional[AffineForm] = None

        if isinstance(abstract_shape, MN_BaB_Shape):  # Single input

            new_lb_form = self._backsub_affine_form(abstract_shape.lb, abstract_shape)

            if abstract_shape.ub is not None:
                new_ub_form = self._backsub_affine_form(
                    abstract_shape.ub, abstract_shape
                )

        else:  # Iterate through all incoming bounds

            assert isinstance(abstract_shape, List)
            assert isinstance(abstract_shape[0], MN_BaB_Shape)

            # new_lb_form_i will have all dimensions properly expanded
            new_lb_form = self._backsub_affine_form(
                abstract_shape[0].lb, abstract_shape[0]
            )
            if abstract_shape[0].ub is not None:
                new_ub_form = self._backsub_affine_form(
                    abstract_shape[0].ub, abstract_shape[0]
                )

            for abs_shape in abstract_shape[1:]:
                assert isinstance(abs_shape, MN_BaB_Shape)
                new_lb_form_i = self._backsub_affine_form(abs_shape.lb, abs_shape)
                new_lb_form.coef += new_lb_form_i.coef
                new_lb_form.bias += new_lb_form_i.bias

                new_ub_form_i: Optional[AffineForm] = None
                if abs_shape.ub is not None:
                    assert new_ub_form is not None
                    new_ub_form_i = self._backsub_affine_form(abs_shape.ub, abs_shape)
                    new_ub_form.coef += new_ub_form_i.coef
                    new_ub_form.bias += new_ub_form_i.bias

            abstract_shape = abstract_shape[0]
        abstract_shape.update_bounds(new_lb_form, new_ub_form)
        return abstract_shape

    def _backsub_affine_form(
        self, affine_form: AffineForm, abstract_shape: MN_BaB_Shape
    ) -> AffineForm:
        if abstract_shape.uses_dependence_sets():
            assert False, "Not implemented - Slice with dependence sets"
        assert isinstance(affine_form.coef, Tensor)

        bs, num_queries = affine_form.coef.shape[:2]
        new_coef_shape = (bs, num_queries, *self.input_dim)

        slice_indices = [
            slice(0, dim, 1) for dim in new_coef_shape
        ]  # list of slices selecting whole input
        slice_indices[self.abs_dim + 2] = slice(
            self.starts, self.ends, self.steps
        )  # replace slice dimension with right stride

        new_coef = torch.zeros(new_coef_shape, device=affine_form.device)
        new_coef[slice_indices] = affine_form.coef
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
        index = torch.tensor(
            range(self.starts, self.ends, self.steps), device=interval_lb.device
        )
        output_lb = torch.index_select(interval_lb, self.abs_dim + 1, index)
        output_ub = torch.index_select(interval_ub, self.abs_dim + 1, index)

        return output_lb, output_ub

    def propagate_abstract_element(
        self,
        abs_input: AbstractElement,
        use_existing_bounds: Optional[bool] = None,
        activation_layer_only: bool = False,
        set_input: bool = True,
        set_output: bool = True,
    ) -> AbstractElement:
        return abs_input.slice(self.abs_dim + 1, self.starts, self.ends, self.steps)
