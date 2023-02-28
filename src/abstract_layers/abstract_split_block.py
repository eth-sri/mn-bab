from __future__ import annotations

from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from src.abstract_domains.ai_util import AbstractElement
from src.abstract_layers.abstract_container_module import (
    AbstractContainerModule,
    ActivationLayer,
)
from src.abstract_layers.abstract_sequential import Sequential
from src.concrete_layers.split_block import SplitBlock as concreteSplitBlock
from src.mn_bab_shape import AffineForm, MN_BaB_Shape
from src.state.subproblem_state import SubproblemState
from src.state.tags import LayerTag, query_tag
from src.utilities.config import BacksubstitutionConfig
from src.utilities.general import tensor_reduce
from src.utilities.queries import get_output_bound_initial_query_coef


class SplitBlock(concreteSplitBlock, AbstractContainerModule):
    def __init__(
        self,
        center_path: nn.Sequential,
        split: Tuple[bool, Tuple[int, ...], Optional[int], int, bool],
        inner_reduce: Tuple[int, bool, bool],
        outer_reduce: Tuple[int, bool, bool],
        input_dim: Tuple[int, ...],
        **kwargs: Any,
    ) -> None:
        super(SplitBlock, self).__init__(
            center_path=center_path,
            split=split,
            inner_reduce=inner_reduce,
            outer_reduce=outer_reduce,
        )

        # Handle all dims
        self.input_dim = input_dim
        self.split = split
        self.split_dim = split[3]
        self.inner_reduce_dim = inner_reduce[0]
        self.outer_reduce_dim = outer_reduce[0]

        # Remove batch from dim
        if self.inner_reduce_dim > 0:
            self.inner_reduce_dim -= 1
        if self.outer_reduce_dim > 0:
            self.outer_reduce_dim -= 1
        if self.split_dim > 0:
            self.split_dim -= 1
        if self.split_dim < 0:
            self.split_dim = len(input_dim) + self.split_dim

        # Dimensions after the split
        interm_dim = list(self.input_dim)
        interm_dim[self.split_dim] = split[1][0]
        self.center_dim = tuple(interm_dim)

        interm_dim = list(self.input_dim)
        interm_dim[self.split_dim] = split[1][1]
        self.res_dim = tuple(interm_dim)

        # Center path
        # mypy doesnt see that the center_path is a subclass of the center_path of the concrete_path
        self.abs_center_path = Sequential.from_concrete_module(  # type: ignore[assignment]
            center_path, self.center_dim, **kwargs
        )

        # Other parameters
        # Box concretization of division factor - set via propagate-interval
        self.res_lower: Optional[Tensor] = None
        self.res_upper: Optional[Tensor] = None

        # Output dimensions
        center_out_dim = list(self.abs_center_path.output_dim)
        self.center_out_dim_pre_reduce = center_out_dim[self.inner_reduce_dim]
        # center_out_dim[self.inner_reduce_dim] = 1
        center_out_dim = [
            dim for (i, dim) in enumerate(center_out_dim) if i != self.inner_reduce_dim
        ]
        self.output_dim = tuple(center_out_dim)

        self.bias = self.get_babsr_bias()

    @classmethod
    def from_concrete_module(  # type: ignore[override] # (checked at runtime)
        cls,
        module: concreteSplitBlock,
        input_dim: Tuple[int, ...],
        **kwargs: Any,
    ) -> SplitBlock:
        assert isinstance(module, concreteSplitBlock)
        abstract_layer = cls(
            module.center_path,
            module.split,
            module.inner_reduce,
            module.outer_reduce,
            input_dim,
            **kwargs,
        )
        return abstract_layer

    def backsubstitute_shape(
        self,
        config: BacksubstitutionConfig,
        input_lb: Tensor,
        input_ub: Tensor,
        abstract_shape: MN_BaB_Shape,
        from_layer_index: Optional[int],
        propagate_preceeding_callback: Optional[
            Callable[
                [BacksubstitutionConfig, MN_BaB_Shape, bool],
                Tuple[MN_BaB_Shape, Tuple[Tensor, Tensor]],
            ]
        ] = None,
        preceeding_layers: Optional[List[Any]] = None,
        use_early_termination_for_current_query: bool = False,
        full_back_prop: bool = False,
        optimize_intermediate_bounds: bool = False,
    ) -> Tuple[MN_BaB_Shape, Tuple[Tensor, Tensor]]:

        assert from_layer_index is None

        assert self.res_lower is not None and self.res_upper is not None
        # Dimensions
        coef_split_dim = self.split_dim + 2

        # Wraps the prceeding callback such that we correctly apply the split header block
        preceeding_callback = self._get_split_block_callback(
            propagate_preceeding_callback
        )

        # Preparation:
        # 1. We start by backpropagating a dummy shape through the center path. This gives us easy access to the output bounds of the path
        # 2. We use these bounds in the abstract mul transformer
        #   - Note that these bounds are all positive i.e. our mul transformer is as tight as it can be
        #   - We build: 1. a backsubstitution matrix for the mul transformer
        #             2. Input lower and upper bounds for the input into the div transformer
        # 3. Use 2.2 and the mul transformer to compute the backprop matrix for the div
        # 4. Backprop through mul
        # 5. Backprop through center

        # Get the output bounds of the center path
        center_path_out_lb, center_path_out_ub = self.get_center_path_out_bounds(
            input_lb,
            input_ub,
            abstract_shape,
            config,
            preceeding_callback,
            preceeding_layers,
        )
        # last_centre_layer = self.abs_center_path.layers[-1]
        # if last_centre_layer.input_bounds is not None and isinstance(
        #     last_centre_layer, ReLU
        # ):
        #     print(
        #         (
        #             F.relu(last_centre_layer.input_bounds[0])
        #             <= F.relu(last_centre_layer.input_bounds[1]) + 1e-7
        #         ).all()
        #     )
        #     center_path_out_lb = F.relu(last_centre_layer.input_bounds[0])
        #     center_path_out_ub = F.relu(last_centre_layer.input_bounds[1])
        # else:
        #     center_path_out_lb, center_path_out_ub = self.get_center_path_out_bounds(
        #         input_lb,
        #         input_ub,
        #         abstract_shape,
        #         config,
        #         preceeding_callback,
        #         preceeding_layers,
        #     )
        #     if isinstance(last_centre_layer, ReLU):
        #         assert last_centre_layer.input_bounds is not None
        #         center_path_out_lb = torch.maximum(
        #             center_path_out_lb, F.relu(last_centre_layer.input_bounds[0])
        #         )
        #         center_path_out_ub = torch.minimum(
        #             center_path_out_ub, F.relu(last_centre_layer.input_bounds[1])
        #         )

        assert (center_path_out_lb <= center_path_out_ub + 1e-10).all()
        # Get the lower and upper-bound slopes and offsets for the multiplication
        res_lower, res_upper = self.res_lower, self.res_upper
        mul_factors = (res_lower, res_upper)
        mul_convex_bounds = self._get_multiplication_slopes_and_intercepts(
            mul_factors, (center_path_out_lb, center_path_out_ub)
        )

        # Get the input bounds for the dividend

        (
            div_input_lb_lb,
            div_input_lb_ub,
            div_input_ub_lb,
            div_input_ub_ub,
        ) = self._get_mul_lbs_and_ubs(
            mul_factors, (center_path_out_lb, center_path_out_ub)
        )
        div_input_lb = torch.minimum(div_input_lb_lb, div_input_ub_lb).sum(
            dim=self.outer_reduce_dim + 1
        )
        div_input_ub = torch.maximum(div_input_lb_ub, div_input_ub_ub).sum(
            dim=self.outer_reduce_dim + 1
        )
        div_input_bounds = (
            div_input_lb,
            div_input_ub,
        )

        # Get the lower and upper-bound slopes and offsets for the division
        div_factor_lower = 1 / res_lower.sum(dim=self.outer_reduce_dim + 1)
        div_factor_upper = 1 / res_lower.sum(dim=self.outer_reduce_dim + 1)
        assert (div_factor_lower * div_factor_upper > 0).all()
        div_factors = (
            torch.minimum(div_factor_lower, div_factor_upper),
            torch.maximum(div_factor_lower, div_factor_upper),
        )
        div_convex_bounds = self._get_multiplication_slopes_and_intercepts(
            div_factors, div_input_bounds
        )

        # Backpropagation Part 1 Div-Reshape
        lower_form = self._backsub_affine_form_first(
            abstract_shape.lb, div_convex_bounds, False, abstract_shape
        )
        upper_form: Optional[AffineForm] = None
        if abstract_shape.ub is not None:
            upper_form = self._backsub_affine_form_first(
                abstract_shape.ub, div_convex_bounds, True, abstract_shape
            )

        # Update Abstract Shape so that we can go through mul layer
        abstract_shape.update_bounds(lower_form, upper_form)

        # Backprop Part 2 - Mul
        lower_form = self._backsub_affine_form_given_convex_bounds(
            abstract_shape.lb, mul_convex_bounds, False, abstract_shape
        )
        if abstract_shape.ub is not None:
            upper_form = self._backsub_affine_form_given_convex_bounds(
                abstract_shape.ub, mul_convex_bounds, True, abstract_shape
            )

        # Update Abstract Shape
        abstract_shape.update_bounds(lower_form, upper_form)

        unstable_queries_old_for_assert = abstract_shape.unstable_queries

        # Backprop center_path
        (center_shape, (lbs_c, ubs_c),) = self.abs_center_path.backsubstitute_shape(
            config,
            input_lb,
            input_ub,
            abstract_shape,
            None,
            preceeding_callback,  # Append the Split-Block callback
            preceeding_layers,  # Append the Split-Block layer
            use_early_termination_for_current_query=False,
            full_back_prop=False,  # Only want to backprop the path
            optimize_intermediate_bounds=optimize_intermediate_bounds,
        )
        assert (
            abstract_shape.unstable_queries is None
            or (
                abstract_shape.unstable_queries == unstable_queries_old_for_assert
            ).all()
        )

        # Backprop through the split
        # As we concretized the second split, we simply append it with 0 sensitivity
        # NOTE: Not generalized for arbitrary splits (assumes only 2 splits)
        assert len(self.split[1]) == 2
        assert isinstance(center_shape.lb.coef, Tensor)
        zero_append_shape = [
            center_shape.lb.coef.shape[0],
            center_shape.lb.coef.shape[1],
            *self.input_dim,
        ]
        zero_append_shape[coef_split_dim] = self.split[1][1]

        zero_append_matrix = torch.zeros(
            zero_append_shape, device=abstract_shape.device
        )
        zero_appended_lb = torch.cat(
            (center_shape.lb.coef, zero_append_matrix), dim=coef_split_dim
        )

        lower_form = AffineForm(zero_appended_lb, center_shape.lb.bias)

        if center_shape.ub is not None:
            assert isinstance(center_shape.ub.coef, Tensor)
            zero_appended_ub = torch.cat(
                (center_shape.ub.coef, zero_append_matrix), dim=coef_split_dim
            )
            upper_form = AffineForm(zero_appended_ub, center_shape.ub.bias)

        abstract_shape.update_bounds(lower_form, upper_form)
        return (
            abstract_shape,
            (
                -np.inf * torch.ones_like(lbs_c, device=abstract_shape.device),
                np.inf * torch.ones_like(ubs_c, device=abstract_shape.device),
            ),  # TODO: this seems unnecessary, move bounds into abstract_shape and just update them when it makes sense
        )

    def _backsub_affine_form_first(
        self,
        affine_form: AffineForm,
        div_convex_bounds: Tuple[Tensor, Tensor, Tensor, Tensor],
        compute_upper_bound: bool,
        abstract_shape: MN_BaB_Shape,
    ) -> AffineForm:

        coef_inner_reduce_dim = self.inner_reduce_dim + 2

        div_form = self._backsub_affine_form_given_convex_bounds(
            affine_form, div_convex_bounds, compute_upper_bound, abstract_shape
        )

        assert isinstance(div_form.coef, Tensor)
        # Backprop through reduce_sum
        repeat_dims = [1] * (len(div_form.coef.shape) + 1)
        repeat_dims[coef_inner_reduce_dim] = self.center_out_dim_pre_reduce
        pre_red_lb_coef = div_form.coef.unsqueeze(coef_inner_reduce_dim).repeat(
            repeat_dims
        )

        return AffineForm(pre_red_lb_coef, div_form.bias)

    def _backsub_affine_form_given_convex_bounds(
        self,
        affine_form: AffineForm,
        convex_bounds: Tuple[Tensor, Tensor, Tensor, Tensor],
        compute_upper_bound: bool,
        abstract_shape: MN_BaB_Shape,
    ) -> AffineForm:
        lb_slope, lb_offset, ub_slope, ub_offset = convex_bounds
        lb_slope = lb_slope.unsqueeze(1)
        lb_offset = lb_offset.unsqueeze(1)
        ub_slope = ub_slope.unsqueeze(1)
        ub_offset = ub_offset.unsqueeze(1)

        # Handle bias
        lb_bias, ub_bias = abstract_shape._matmul_of_coef_and_interval(
            lb_offset, ub_offset
        )

        new_bias = ub_bias if compute_upper_bound else lb_bias
        assert new_bias is not None

        new_bias += affine_form.bias

        # Handle coef
        new_coef: Optional[Tensor]
        new_lb_coef, new_ub_coef = abstract_shape._elementwise_mul_of_coef_and_interval(
            lb_slope, ub_slope
        )
        new_coef = new_ub_coef if compute_upper_bound else new_lb_coef
        assert new_coef is not None

        return AffineForm(new_coef, new_bias)

    def get_babsr_bias(self) -> Tensor:
        return self.abs_center_path.get_babsr_bias()

    def reset_input_bounds(self) -> None:
        super(SplitBlock, self).reset_input_bounds()
        self.abs_center_path.reset_input_bounds()

    def reset_optim_input_bounds(self) -> None:
        super(SplitBlock, self).reset_input_bounds()
        self.abs_center_path.reset_optim_input_bounds()

    def reset_output_bounds(self) -> None:
        super(SplitBlock, self).reset_output_bounds()
        self.abs_center_path.reset_output_bounds()

    def propagate_interval(
        self,
        interval: Tuple[Tensor, Tensor],
        use_existing_bounds: Optional[bool] = None,
        subproblem_state: Optional[SubproblemState] = None,
        activation_layer_only: bool = False,
        set_input: bool = True,
        set_output: bool = True,
    ) -> Tuple[Tensor, Tensor]:

        center_lower, res_lower = torch.split(
            interval[0], split_size_or_sections=self.split[1], dim=self.split_dim + 1
        )
        center_upper, res_upper = torch.split(
            interval[1], split_size_or_sections=self.split[1], dim=self.split_dim + 1
        )

        lower_out, upper_out = self.abs_center_path.propagate_interval(
            (center_lower, center_upper),
            use_existing_bounds=use_existing_bounds,
            subproblem_state=subproblem_state,
            activation_layer_only=activation_layer_only,
            set_input=set_input,
            set_output=set_output,
        )

        all_combs = [
            lower_out * res_lower,
            lower_out * res_upper,
            upper_out * res_lower,
            upper_out * res_upper,
        ]
        lower_inner_merge = tensor_reduce(torch.minimum, all_combs)
        upper_inner_merge = tensor_reduce(torch.maximum, all_combs)

        lower_inner_reduce = torch.sum(
            lower_inner_merge, dim=self.inner_reduce_dim + 1
        )  # We propagate with batch size
        upper_inner_reduce = torch.sum(upper_inner_merge, dim=self.inner_reduce_dim + 1)

        lower_outer_reduce = torch.sum(res_lower, dim=self.outer_reduce_dim + 1)
        upper_outer_reduce = torch.sum(res_upper, dim=self.outer_reduce_dim + 1)

        # If the interval contains 0 we would have NaNs
        assert (lower_outer_reduce * upper_outer_reduce > 0).all()

        # Save this for backward pass
        self.res_lower = res_lower
        self.res_upper = res_upper

        all_combs = [
            lower_inner_reduce / lower_outer_reduce,
            lower_inner_reduce / upper_outer_reduce,
            upper_inner_reduce / lower_outer_reduce,
            upper_inner_reduce / upper_outer_reduce,
        ]

        lower_out = tensor_reduce(torch.minimum, all_combs)
        upper_out = tensor_reduce(torch.maximum, all_combs)

        return (lower_out, upper_out)

    def propagate_abstract_element(
        self,
        abs_input: AbstractElement,
        use_existing_bounds: Optional[bool] = None,
        activation_layer_only: bool = False,
        set_input: bool = True,
        set_output: bool = True,
    ) -> AbstractElement:

        center_abs, res_abs = abs_input.split(
            split_size_or_sections=self.split[1], dim=self.split_dim + 1
        )

        res_lb, res_ub = res_abs.concretize()

        center_out = self.abs_center_path.propagate_abstract_element(
            center_abs,
            use_existing_bounds=use_existing_bounds,
            activation_layer_only=activation_layer_only,
            set_input=set_input,
            set_output=set_output,
        )

        # Multiplication
        mul_ae = center_out.multiply_interval((res_lb, res_ub))

        # Inner reduction
        inner_ae = mul_ae.sum(self.inner_reduce_dim + 1, reduce_dim=True)

        # Outer reduction
        res_lb_reduce = torch.sum(res_lb, dim=self.outer_reduce_dim + 1)
        res_ub_reduce = torch.sum(res_ub, dim=self.outer_reduce_dim + 1)

        # If the interval contains 0 we would have NaNs
        assert (res_lb_reduce * res_ub_reduce > 0).all()

        # Division

        div_factors = (
            1 / torch.maximum(res_ub_reduce, res_lb_reduce),
            1 / torch.minimum(res_ub_reduce, res_lb_reduce),
        )
        assert (div_factors[0] <= div_factors[1]).all()
        out_ae = inner_ae.multiply_interval(div_factors)

        return out_ae

    def set_dependence_set_applicability(self, applicable: bool = True) -> None:
        self.abs_center_path.set_dependence_set_applicability(applicable)
        self.dependence_set_applicable = self.abs_center_path.layers[
            -1
        ].dependence_set_applicable

    def get_default_split_constraints(
        self, batch_size: int, device: torch.device
    ) -> Dict[LayerTag, Tensor]:
        split_constraints: Dict[LayerTag, Tensor] = {}
        split_constraints.update(
            self.abs_center_path.get_default_split_constraints(batch_size, device)
        )
        return split_constraints

    def get_default_split_points(
        self, batch_size: int, device: torch.device
    ) -> Dict[LayerTag, Tensor]:
        split_points: Dict[LayerTag, Tensor] = {}
        split_points.update(
            self.abs_center_path.get_default_split_points(batch_size, device)
        )
        return split_points

    def get_activation_layers(self) -> Dict[LayerTag, ActivationLayer]:
        act_layers: Dict[LayerTag, ActivationLayer] = {}
        act_layers.update(self.abs_center_path.get_activation_layers())
        return act_layers

    def get_current_intermediate_bounds(
        self,
    ) -> OrderedDict[LayerTag, Tuple[Tensor, Tensor]]:
        intermediate_bounds: OrderedDict[
            LayerTag, Tuple[Tensor, Tensor]
        ] = OrderedDict()
        intermediate_bounds.update(
            self.abs_center_path.get_current_intermediate_bounds()
        )
        return intermediate_bounds

    def get_current_optimized_intermediate_bounds(
        self,
    ) -> OrderedDict[LayerTag, Tuple[Tensor, Tensor]]:
        intermediate_bounds: OrderedDict[
            LayerTag, Tuple[Tensor, Tensor]
        ] = OrderedDict()
        intermediate_bounds.update(
            self.abs_center_path.get_current_optimized_intermediate_bounds()
        )
        return intermediate_bounds

    def set_intermediate_input_bounds(
        self, intermediate_bounds: OrderedDict[LayerTag, Tuple[Tensor, Tensor]]
    ) -> None:
        self.abs_center_path.set_intermediate_input_bounds(intermediate_bounds)

    def get_activation_layer_ids(
        self, act_layer_ids: Optional[List[LayerTag]] = None
    ) -> List[LayerTag]:
        act_layer_ids = self.abs_center_path.get_activation_layer_ids()
        return act_layer_ids

    def get_relu_layer_ids(
        self, act_layer_ids: Optional[List[LayerTag]] = None
    ) -> List[LayerTag]:
        act_layer_ids = self.abs_center_path.get_relu_layer_ids()
        return act_layer_ids

    def get_center_path_out_bounds(
        self,
        input_lb: Tensor,
        input_ub: Tensor,
        abstract_shape: MN_BaB_Shape,
        config: BacksubstitutionConfig,
        preceeding_callback: Optional[
            Callable[
                [BacksubstitutionConfig, MN_BaB_Shape, bool],
                Tuple[MN_BaB_Shape, Tuple[Tensor, Tensor]],
            ]
        ] = None,
        preceeding_layers: Optional[List[Any]] = None,
    ) -> Tuple[Tensor, Tensor]:

        intermediate_bounds_to_recompute = None  # compute all out bounds

        initial_intermediate_bound_coef = get_output_bound_initial_query_coef(
            dim=self.abs_center_path.output_dim,
            intermediate_bounds_to_recompute=intermediate_bounds_to_recompute,
            batch_size=abstract_shape.batch_size,
            use_dependence_sets=config.use_dependence_sets,
            device=abstract_shape.device,
            dtype=None,
        )

        center_bound_abstract_shape = MN_BaB_Shape(  # Here AffineForm will be cloned later
            query_id=query_tag(self.abs_center_path),
            query_prev_layer=None,  # TODO: do we want reduced parameter sharing for those bounds?
            queries_to_compute=intermediate_bounds_to_recompute,
            lb=AffineForm(initial_intermediate_bound_coef),
            ub=AffineForm(initial_intermediate_bound_coef),
            unstable_queries=None,  # (not using early termination)
            subproblem_state=abstract_shape.subproblem_state,
        )

        (
            propagated_shape,
            layer_bounds,
        ) = self.abs_center_path._get_mn_bab_shape_after_layer(
            from_layer_index=len(self.abs_center_path.layers)
            - 1,  # Full Backprop through center layer
            config=config.where(use_early_termination=False),
            input_lb=input_lb,
            input_ub=input_ub,
            abstract_shape=center_bound_abstract_shape,
            propagate_preceeding_callback=preceeding_callback,
            preceeding_layers=preceeding_layers,
            use_early_termination_for_current_query=False,  # TODO why not?
            optimize_intermediate_bounds=False,
        )

        assert propagated_shape is not None
        assert layer_bounds is None
        (
            center_path_out_lb,
            center_path_out_ub,
        ) = propagated_shape.concretize(input_lb, input_ub)

        center_path_out_lb = center_path_out_lb.view_as(
            self.abs_center_path.layers[-1].input_bounds[0]
        )
        assert center_path_out_ub is not None
        center_path_out_ub = center_path_out_ub.view_as(
            self.abs_center_path.layers[-1].input_bounds[1]
        )

        return (center_path_out_lb, center_path_out_ub)

    def _get_multiplication_slopes_and_intercepts(
        self, mul_bounds: Tuple[Tensor, Tensor], input_bounds: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:

        input_lb, input_ub = input_bounds

        # Get the lower and upper bound of the multiplication
        (mult_lb_lb, mult_lb_ub, mult_ub_lb, mult_ub_ub) = self._get_mul_lbs_and_ubs(
            mul_bounds, input_bounds
        )

        D = 1e-12 if input_lb.dtype == torch.float64 else 1e-7

        # Get slopes and offsets
        # TODO look at effect of soundness correction here
        convex_lb_slope = (mult_ub_lb - mult_lb_lb) / (input_ub - input_lb + D)
        convex_lb_intercept = mult_lb_lb - input_lb * convex_lb_slope - D

        convex_ub_slope = (mult_ub_ub - mult_lb_ub) / (input_ub - input_lb + D)
        convex_ub_intercept = mult_lb_ub - input_lb * convex_ub_slope + D

        return (
            convex_lb_slope,
            convex_lb_intercept,
            convex_ub_slope,
            convex_ub_intercept,
        )

    def _get_mul_lbs_and_ubs(
        self, b1: Tuple[Tensor, Tensor], b2: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        input_lb_opts = [b2[0] * b1[0], b2[0] * b1[1]]
        input_ub_opts = [b2[1] * b1[0], b2[1] * b1[1]]
        mult_lb_lb = tensor_reduce(torch.minimum, input_lb_opts)
        mult_lb_ub = tensor_reduce(torch.maximum, input_lb_opts)
        mult_ub_lb = tensor_reduce(torch.minimum, input_ub_opts)
        mult_ub_ub = tensor_reduce(torch.maximum, input_ub_opts)
        return (mult_lb_lb, mult_lb_ub, mult_ub_lb, mult_ub_ub)

    def _get_split_block_callback(
        self,
        propagate_preceeding_callback: Optional[
            Callable[
                [BacksubstitutionConfig, MN_BaB_Shape, bool],
                Tuple[MN_BaB_Shape, Tuple[Tensor, Tensor]],
            ]
        ],
    ) -> Callable[
        [BacksubstitutionConfig, MN_BaB_Shape, bool],
        Tuple[MN_BaB_Shape, Tuple[Tensor, Tensor]],
    ]:
        """ReLU layers within the center path need a propagate preceeding callback that takes the split at the top into account"""

        def wrapped_call(
            config: BacksubstitutionConfig,
            abstract_shape: MN_BaB_Shape,
            use_early_termination_for_current_query: bool,
        ) -> Tuple[MN_BaB_Shape, Tuple[Tensor, Tensor]]:
            # Backwards prop through split-block
            coef_split_dim = self.split_dim + 2
            lb = abstract_shape.lb

            assert isinstance(lb.coef, Tensor)

            zero_append_shape = [lb.coef.shape[0], lb.coef.shape[1], *self.input_dim]
            zero_append_shape[coef_split_dim] = self.split[1][1]

            zero_append_matrix = torch.zeros(
                zero_append_shape, device=abstract_shape.device
            )
            zero_appended_lb = torch.cat(
                (lb.coef, zero_append_matrix), dim=coef_split_dim
            ).to(abstract_shape.device)

            lower_form = AffineForm(zero_appended_lb, lb.bias)
            upper_form: Optional[AffineForm] = None

            if abstract_shape.ub is not None:
                ub = abstract_shape.ub
                assert isinstance(ub.coef, Tensor)
                zero_appended_ub = torch.cat(
                    (ub.coef, zero_append_matrix), dim=coef_split_dim
                ).to(abstract_shape.device)
                upper_form = AffineForm(zero_appended_ub, ub.bias)

            abstract_shape.update_bounds(lower_form, upper_form)

            if propagate_preceeding_callback is None:
                assert isinstance(abstract_shape.lb.coef, Tensor)
                bound_shape = abstract_shape.lb.coef.shape[:2]
                return (
                    abstract_shape,
                    (
                        -np.inf * torch.ones(bound_shape, device=abstract_shape.device),
                        np.inf * torch.ones(bound_shape, device=abstract_shape.device),
                    ),  # TODO: this seems unnecessary, move bounds into abstract_shape and just update them when it makes sense
                )
            else:
                return propagate_preceeding_callback(
                    config,
                    abstract_shape,
                    use_early_termination_for_current_query,
                )

        return wrapped_call
