from __future__ import annotations

import typing
from typing import Any, Optional, Sequence, Tuple, Union

from torch import Tensor

from src.abstract_domains.ai_util import AbstractElement
from src.abstract_layers.abstract_module import AbstractModule
from src.concrete_layers.binary_op import BinaryOp as concreteBinaryOp
from src.mn_bab_shape import AffineForm, MN_BaB_Shape
from src.state.subproblem_state import SubproblemState
from src.utilities.config import BacksubstitutionConfig
from src.utilities.dependence_sets import DependenceSets


class BinaryOp(concreteBinaryOp, AbstractModule):
    def __init__(self, op: str, input_dim: Tuple[int, ...]) -> None:
        super(BinaryOp, self).__init__(op)
        self.op = op
        self.input_dim = input_dim
        self.output_dim = input_dim

    @classmethod
    def from_concrete_module(  # type: ignore[override] # (checked at runtime)
        cls,
        module: concreteBinaryOp,
        input_dim: Sequence[Tuple[int, ...]],
        **kwargs: Any,
    ) -> BinaryOp:
        assert isinstance(module, concreteBinaryOp)
        assert len(input_dim) == 2
        assert input_dim[0] == input_dim[1]
        return cls(module.op, input_dim[0])

    def backsubstitute(  # type:ignore[override]
        self,
        config: BacksubstitutionConfig,
        abstract_shape: MN_BaB_Shape,
    ) -> Tuple[MN_BaB_Shape, MN_BaB_Shape]:

        new_lb_left_form, new_lb_right_form = self._backsub_affine_form(
            abstract_shape.lb, abstract_shape
        )
        new_ub_left_form: Optional[AffineForm] = None
        new_ub_right_form: Optional[AffineForm] = None
        if abstract_shape.ub is not None:
            new_ub_left_form, new_ub_right_form = self._backsub_affine_form(
                abstract_shape.ub, abstract_shape
            )

        left_as = abstract_shape.clone_with_new_bounds(
            new_lb_left_form, new_ub_left_form
        )  # redundant
        right_as = abstract_shape.clone_with_new_bounds(
            new_lb_right_form, new_ub_right_form
        )
        return (left_as, right_as)

    def _backsub_affine_form(
        self, affine_form: AffineForm, abstract_shape: MN_BaB_Shape
    ) -> Tuple[AffineForm, AffineForm]:
        if abstract_shape.uses_dependence_sets():
            assert isinstance(affine_form.coef, DependenceSets)
            coef = affine_form.coef.sets
        else:
            assert isinstance(affine_form.coef, Tensor)
            coef = affine_form.coef

        left_coef = coef
        right_coef = coef

        bias = affine_form.bias

        if self.op == "add":
            pass
        elif self.op == "sub":
            right_coef = -1 * right_coef
        else:
            assert False, f"Unknown operator {self.op}"

        final_left_coef: Union[Tensor, DependenceSets] = left_coef
        final_right_coef: Union[Tensor, DependenceSets] = right_coef
        if abstract_shape.uses_dependence_sets():
            assert isinstance(affine_form.coef, DependenceSets)
            assert isinstance(final_left_coef, Tensor)
            assert isinstance(final_right_coef, Tensor)
            final_left_coef = DependenceSets(
                final_left_coef,
                affine_form.coef.spatial_idxs,
                affine_form.coef.input_dim,
                affine_form.coef.cstride,
                affine_form.coef.cpadding,
            )
            final_right_coef = DependenceSets(
                final_right_coef,
                affine_form.coef.spatial_idxs,
                affine_form.coef.input_dim,
                affine_form.coef.cstride,
                affine_form.coef.cpadding,
            )

        return (AffineForm(final_left_coef, bias), AffineForm(final_right_coef, bias))

    @typing.no_type_check  # Mypy can't handle the buffer type
    def propagate_interval(
        self,
        intervals: Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]],
        use_existing_bounds: Optional[bool] = None,
        subproblem_state: Optional[SubproblemState] = None,
        activation_layer_only: bool = False,
        set_input: bool = True,
        set_output: bool = True,
    ) -> Tuple[Tensor, Tensor]:

        left_lb, left_ub = intervals[0]
        right_lb, right_ub = intervals[1]

        if self.op == "add":
            return (left_lb + right_lb, left_ub + right_ub)
        elif self.op == "sub":
            return (left_lb - right_ub, left_ub - right_lb)
        else:
            assert False, f"Unknown operation {self.op}"

    def propagate_abstract_element(  # type: ignore [override] # supertype expects just one abstract element, but this is a special case
        self,
        abs_inputs: Tuple[AbstractElement, AbstractElement],
        use_existing_bounds: Optional[bool] = None,
        activation_layer_only: bool = False,
        set_input: bool = True,
        set_output: bool = True,
    ) -> AbstractElement:

        if self.op == "add":
            return abs_inputs[0] + abs_inputs[1]
        elif self.op == "sub":
            return abs_inputs[0] - abs_inputs[1]
        else:
            assert False, f"Unknown operation {self.op}"
