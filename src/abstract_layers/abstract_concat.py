from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from src.abstract_domains.ai_util import AbstractElement
from src.abstract_layers.abstract_module import AbstractModule
from src.concrete_layers.concat import Concat as concreteConcat
from src.mn_bab_shape import AffineForm, MN_BaB_Shape
from src.state.subproblem_state import SubproblemState
from src.utilities.config import BacksubstitutionConfig


class Concat(concreteConcat, AbstractModule):
    def __init__(
        self,
        dim: int,
        input_dims: Sequence[Tuple[int, ...]],
    ):
        super(Concat, self).__init__(dim)
        self.abs_dim = dim - 1  # no batch dimension

        self.input_dims = input_dims  # Ordered list of incoming dimensions
        cat_dim = sum([input_dim[self.abs_dim] for input_dim in input_dims])
        # Should assert that all dims otherwise are equal
        for input_dim in input_dims:
            for i in range(len(input_dims[0])):
                if i != self.abs_dim:
                    assert (
                        input_dim[i] == input_dims[0][i]
                    ), f"Dimension mismatch in concat input: {input_dim} {input_dims[0]}"

        output_dim: list[int] = list(input_dim)
        output_dim[self.abs_dim] = cat_dim
        self.output_dim = tuple(output_dim)

    @classmethod
    def from_concrete_module(  # type: ignore[override] # (checked at runtime)
        cls,
        module: concreteConcat,
        input_dims: Sequence[Tuple[int, ...]],
        **kwargs: Any,
    ) -> Concat:
        assert isinstance(module, concreteConcat)
        abstract_layer = cls(module.dim, input_dims)
        return abstract_layer

    def backsubstitute(  # type: ignore[override]
        self, config: BacksubstitutionConfig, abstract_shape: MN_BaB_Shape
    ) -> Sequence[MN_BaB_Shape]:

        shapes: List[MN_BaB_Shape] = []

        offset = 0
        for in_shape in self.input_dims:
            new_lb_form = self._backsub_affine_form(
                abstract_shape.lb, abstract_shape, offset, in_shape
            )
            new_ub_form: Optional[AffineForm] = None
            if abstract_shape.ub is not None:
                new_ub_form = self._backsub_affine_form(
                    abstract_shape.ub, abstract_shape, offset, in_shape
                )
            offset += in_shape[self.abs_dim]
            new_as = abstract_shape.clone_with_new_bounds(new_lb_form, new_ub_form)
            shapes.append(new_as)

        return shapes

    def _backsub_affine_form(
        self,
        affine_form: AffineForm,
        abstract_shape: MN_BaB_Shape,
        offset: int,
        in_shape: Tuple[int, ...],
    ) -> AffineForm:
        if abstract_shape.uses_dependence_sets():
            assert False, "Not implemented - Concat with dependence sets"
        assert isinstance(affine_form.coef, Tensor)

        qb_dim = (
            self.abs_dim + 2
        )  # Dimension that accounts for query and batch dimension
        # q, b, c, h, w
        # Slice at the correct dimension
        indices = torch.tensor(
            range(offset, offset + in_shape[self.abs_dim]), device=affine_form.device
        )
        new_coef = torch.index_select(affine_form.coef, qb_dim, indices)
        new_bias = affine_form.bias

        return AffineForm(new_coef, new_bias)

    def propagate_interval(  # type: ignore[override]
        self,
        intervals: List[Tuple[Tensor, Tensor]],
        use_existing_bounds: Optional[bool] = None,
        subproblem_state: Optional[SubproblemState] = None,
        activation_layer_only: bool = False,
        set_input: bool = True,
        set_output: bool = True,
    ) -> Tuple[Tensor, Tensor]:

        lbs = [lb for (lb, _) in intervals]
        ubs = [ub for (_, ub) in intervals]
        output_lb = torch.cat(lbs, self.abs_dim + 1)
        output_ub = torch.cat(ubs, self.abs_dim + 1)

        return output_lb, output_ub

    def propagate_abstract_element(  # type: ignore [override]
        self,
        abs_inputs: List[AbstractElement],
        use_existing_bounds: Optional[bool] = None,
        activation_layer_only: bool = False,
        set_input: bool = True,
        set_output: bool = True,
    ) -> AbstractElement:
        return abs_inputs[0].cat(abs_inputs, self.abs_dim + 1)
