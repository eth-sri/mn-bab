from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from src.abstract_layers.abstract_bn2d import BatchNorm2d
from src.abstract_layers.abstract_conv2d import Conv2d
from src.abstract_layers.abstract_linear import Linear
from src.abstract_layers.abstract_module import AbstractModule
from src.abstract_layers.abstract_mulit_path_block import MultiPathBlock
from src.abstract_layers.abstract_network import AbstractNetwork
from src.abstract_layers.abstract_relu import ReLU
from src.abstract_layers.abstract_residual_block import ResidualBlock
from src.abstract_layers.abstract_sequential import Sequential
from src.abstract_layers.abstract_sigmoid import Sigmoid
from src.abstract_layers.abstract_split_block import SplitBlock
from src.abstract_layers.abstract_tanh import Tanh
from src.mn_bab_optimizer import MNBabOptimizer
from src.mn_bab_shape import AffineForm, MN_BaB_Shape
from src.state.parameters import ReadonlyParametersForQuery
from src.state.split_state import ReadonlySplitState
from src.state.subproblem_state import ReadonlySubproblemState, SubproblemState
from src.state.tags import (
    LayerTag,
    NodeTag,
    QueryTag,
    key_alpha_relu_lb,
    key_alpha_relu_ub,
    key_beta_lb,
    key_beta_ub,
    key_prima_lb,
    key_prima_ub,
    layer_from_query_tag,
    layer_tag,
    query_tag,
)
from src.utilities.batching import batch_subproblems
from src.utilities.config import (
    BaBsrBranchingConfig,
    BacksubstitutionConfig,
    BranchingConfig,
    BranchingMethod,
    FilteredSmartBranchingConfig,
    PropagationEffectMode,
    ReduceOp,
    make_branching_config,
)
from src.utilities.general import get_neg_pos_comp
from src.utilities.queries import get_output_bound_initial_query_coef
from src.verification_subproblem import (
    ReadonlyVerificationSubproblem,
    VerificationSubproblem,
)


class SplitIndexFinder:
    network: AbstractNetwork
    backsubstitutionConfig: BacksubstitutionConfig
    query_coef: Tensor
    split_cost_by_layer: Optional[Dict[LayerTag, float]]
    branching_config: BranchingConfig
    # (the following parameters are only used for filtered smart branching)
    input_lb: Tensor
    input_ub: Tensor
    batch_sizes: Sequence[int]
    recompute_intermediate_bounds_after_branching: bool
    optimizer: MNBabOptimizer

    def __init__(
        self,
        network: AbstractNetwork,
        backsubstitution_config: BacksubstitutionConfig,
        query_coef: Tensor,
        split_cost_by_layer: Optional[Dict[LayerTag, float]],
        branching_config: BranchingConfig,
        # (the following parameters are only used for filtered smart branching)
        input_lb: Tensor,
        input_ub: Tensor,
        batch_sizes: Sequence[int],
        recompute_intermediate_bounds_after_branching: bool,
        optimizer: MNBabOptimizer,
    ):
        self.network = network
        self.backsubstitution_config = backsubstitution_config
        self.query_coef = query_coef
        self.split_cost_by_layer = split_cost_by_layer
        self.branching_config = branching_config
        # (the following parameters are only used for filtered smart branching)
        self.input_lb = input_lb
        self.input_ub = input_ub
        self.batch_sizes = batch_sizes
        self.recompute_intermediate_bounds_after_branching = (
            recompute_intermediate_bounds_after_branching
        )
        self.optimizer = optimizer

    def find_node_to_split(self, subproblem: ReadonlyVerificationSubproblem) -> NodeTag:
        return find_node_to_split(
            subproblem,
            self.network,
            self.backsubstitution_config,
            self.query_coef,
            self.split_cost_by_layer,
            self.branching_config,
            # (the following options are only used for filtered smart branching)
            self.input_lb,
            self.input_ub,
            self.batch_sizes,
            self.recompute_intermediate_bounds_after_branching,
            self.optimizer,
        )


def make_split_index_finder(
    network: AbstractNetwork,
    backsubstitution_config: BacksubstitutionConfig,
    query_coef: Tensor,
    initial_subproblem: VerificationSubproblem,
    branching_config: BranchingConfig,
    # (the fonnnnnllowing parameters are only used for filtered smart branching)
    input_lb: Tensor,
    input_ub: Tensor,
    batch_sizes: Sequence[int],
    recompute_intermediate_bounds_after_branching: bool,
    optimizer: MNBabOptimizer,
) -> SplitIndexFinder:
    split_cost_by_layer = None
    if branching_config.use_cost_adjusted_scores:
        assert (
            initial_subproblem.subproblem_state.constraints.prima_constraints
        ), "prima constraints missing with use_cost_adjusted_scores"
        split_cost_by_layer = compute_split_cost_by_layer(
            network,
            initial_subproblem.subproblem_state.constraints.prima_constraints.prima_coefficients,  # TODO: ugly
            recompute_intermediate_bounds_after_branching,
        )
    return SplitIndexFinder(
        network,
        backsubstitution_config,
        query_coef,
        split_cost_by_layer,
        branching_config,
        # (the following parameters are only used for filtered smart branching)
        input_lb,
        input_ub,
        batch_sizes,
        recompute_intermediate_bounds_after_branching,
        optimizer,
    )


def find_node_to_split(
    subproblem: ReadonlyVerificationSubproblem,
    network: AbstractNetwork,
    backsubstitution_config: BacksubstitutionConfig,
    query_coef: Tensor,
    split_cost_by_layer: Optional[Dict[LayerTag, float]],
    branching_config: BranchingConfig,
    # (the following options are only used for filtered smart branching)
    input_lb: Tensor,
    input_ub: Tensor,
    batch_sizes: Sequence[int],
    recompute_intermediate_bounds_after_branching: bool,
    optimizer: MNBabOptimizer,
) -> NodeTag:
    if branching_config.method == BranchingMethod.babsr:
        babsr_config = branching_config.babsr()
        node_to_split = find_index_to_split_with_babsr(
            subproblem,
            network,
            backsubstitution_config,
            query_coef,
            split_cost_by_layer,
            babsr_config,
            False,
        )
    elif branching_config.method == BranchingMethod.active_constraint_score:
        node_to_split = find_index_to_split_with_babsr(
            subproblem,
            network,
            backsubstitution_config,
            query_coef,
            split_cost_by_layer,
            make_branching_config(
                method=BranchingMethod.babsr,
                use_prima_contributions=True,
                use_optimized_slopes=True,
                use_beta_contributions=True,
                propagation_effect_mode=PropagationEffectMode.none,
                use_indirect_effect=False,
                reduce_op=ReduceOp.min,
                use_abs=False,
            ).babsr(),
            True,
        )
    elif branching_config.method == BranchingMethod.filtered_smart_branching:
        filtered_smart_branching_config = branching_config.filtered_smart_branching()
        node_to_split = find_index_to_split_with_filtered_smart_branching(
            subproblem,
            network,
            backsubstitution_config,
            query_coef,
            split_cost_by_layer,
            filtered_smart_branching_config,
            input_lb,
            input_ub,
            batch_sizes,
            recompute_intermediate_bounds_after_branching,
            optimizer,
        )
    else:
        raise RuntimeError("Branching method misspecified.")
    return node_to_split


# adapted from: https://github.com/KaidiXu/Beta-CROWN/blob/master/src/babsr_score_conv.py
# commit hash: 77a055a39bd338367b9c335316004863681fb671
def find_index_to_split_with_babsr(
    subproblem: ReadonlyVerificationSubproblem,
    network: AbstractNetwork,
    backsubstitution_config: BacksubstitutionConfig,
    query_coef: Tensor,
    split_cost_by_layer: Optional[Dict[LayerTag, float]],
    babsr_config: BaBsrBranchingConfig,
    use_active_constraint_branching_scores: bool,
) -> NodeTag:
    assert (
        not subproblem.is_fully_split
    ), "Can't find a node to split for fully split subproblems."
    use_prima_contributions = babsr_config.use_prima_contributions
    assert (
        not use_active_constraint_branching_scores or use_prima_contributions
    ), "Must provide prima contributions for active constraint branching scores"
    use_optimized_slopes = babsr_config.use_optimized_slopes
    use_beta_contributions = babsr_config.use_beta_contributions
    propagation_effect_mode = babsr_config.propagation_effect_mode
    use_indirect_effect = babsr_config.use_indirect_effect
    lower_bound_reduce_op = _get_lower_bound_reduce_op(babsr_config.reduce_op)
    use_abs = babsr_config.use_abs

    # arguments in beta-crown implementation that always set to these values
    decision_threshold = 0.001
    activation_layer_ids = network.get_activation_layer_ids()
    sparsest_layer_id = activation_layer_ids[0]

    device = next(network.parameters()).device
    batch_size = 1

    subproblem_state = (
        subproblem.subproblem_state
        if use_prima_contributions
        else subproblem.subproblem_state.without_prima()
    ).deep_copy_to(
        device
    )  # TODO: avoid copying layer bounds for active constraint score?

    assert (
        not use_prima_contributions
        or subproblem_state.constraints.prima_constraints is not None
    ), "prima constraints missing with use_prima_contributions"

    if use_active_constraint_branching_scores:
        assert subproblem_state.constraints.prima_constraints is not None
        score, intercept_tb = _compute_active_constraint_scores(
            network,
            subproblem_state,
            batch_size,
            device,
        )
    else:
        # NOTE Not yet implemented for non-ReLU activations

        score, intercept_tb = _compute_split_scores(
            backsubstitution_config,
            query_coef,
            network,
            subproblem_state if use_prima_contributions else subproblem_state.without_prima(),
            batch_size,
            device,
            use_optimized_slopes=use_optimized_slopes,
            use_beta_contributions=use_beta_contributions,
            propagation_effect_mode=propagation_effect_mode,
            use_indirect_effect=use_indirect_effect,
            lower_bound_reduce_op=lower_bound_reduce_op,
            use_abs=use_abs,
        )

    assert all(layer_scores.shape[0] == batch_size for layer_scores in score.values())
    if split_cost_by_layer is not None:
        score, intercept_tb = _adjust_based_on_cost(
            score, intercept_tb, split_cost_by_layer
        )

    decision: List[NodeTag] = []
    for batch_index in range(batch_size):
        new_score = {k: score[k][batch_index] for k in score.keys()}
        max_info = {k: torch.max(new_score[k]) for k in new_score.keys()}
        decision_layer_id = sorted(
            new_score.keys(), key=lambda x: float(torch.max(new_score[x])), reverse=True
        )[0]
        decision_index_flattened = torch.argmax(new_score[decision_layer_id])
        decision_index = np.unravel_index(
            decision_index_flattened.cpu(), new_score[decision_layer_id].shape
        )

        if (
            decision_layer_id != sparsest_layer_id
            and max_info[decision_layer_id].item() > decision_threshold
        ):
            decision.append(
                NodeTag(
                    layer=decision_layer_id, index=tuple(int(v) for v in decision_index)
                )
            )
        else:
            new_intercept_tb = {
                k: intercept_tb[k][batch_index] for k in intercept_tb.keys()
            }
            min_info = {
                k: torch.min(new_intercept_tb[k])
                for k in new_intercept_tb.keys()
                if torch.min(new_intercept_tb[k]) < -1e-4
            }

            if len(min_info) != 0:  # and Icp_score_counter < 2:
                intercept_layer_id = [
                    idx for idx in activation_layer_ids if idx in min_info.keys()
                ][-1]
                intercept_index_flattened = torch.argmin(
                    new_intercept_tb[intercept_layer_id]
                )
                intercept_index = np.unravel_index(
                    intercept_index_flattened.cpu(),
                    new_intercept_tb[intercept_layer_id].shape,
                )

                decision.append(
                    NodeTag(
                        layer=intercept_layer_id,
                        index=tuple(int(v) for v in intercept_index),
                    )
                )
            else:
                assert subproblem_state.constraints.split_state is not None
                decision.append(
                    _find_random_node_to_split(
                        subproblem_state.constraints.split_state, batch_index
                    )
                )

    assert len(decision) == batch_size
    return decision[0]


def geo_mean(x: Tensor, y: Tensor) -> Tensor:
    return torch.sqrt(F.relu(x * y))


def _get_lower_bound_reduce_op(
    lower_bound_reduce_op_tag: ReduceOp,
) -> Callable[[Tensor, Tensor], Tensor]:
    lower_bound_reduce_op: Callable[[Tensor, Tensor], Tensor]
    if lower_bound_reduce_op_tag == ReduceOp.min:
        lower_bound_reduce_op = torch.minimum
    elif lower_bound_reduce_op_tag == ReduceOp.max:
        lower_bound_reduce_op = torch.maximum
    elif lower_bound_reduce_op_tag == ReduceOp.geo_mean:
        lower_bound_reduce_op = geo_mean
    else:
        raise RuntimeError("Unknown reduce operation for branching")
    return lower_bound_reduce_op


def _compute_active_constraint_scores(
    network: AbstractNetwork,
    subproblem_state: ReadonlySubproblemState,
    batch_size: int,
    device: torch.device,
) -> Tuple[Dict[LayerTag, Tensor], Dict[LayerTag, Tensor]]:
    score: Dict[LayerTag, Tensor] = {}
    backup_score: Dict[LayerTag, Tensor] = {}

    return _compute_active_constraint_scores_sequential(
        score,
        backup_score,
        network,
        subproblem_state.parameters.parameters_by_query[query_tag(network)],
        subproblem_state,
        batch_size,
        device,
    )


def _compute_active_constraint_scores_sequential(
    score: Dict[LayerTag, Tensor],
    backup_score: Dict[LayerTag, Tensor],
    network: Sequential,
    optimizable_parameters: ReadonlyParametersForQuery,
    subproblem_state: ReadonlySubproblemState,
    batch_size: int,
    device: torch.device,
) -> Tuple[Dict[LayerTag, Tensor], Dict[LayerTag, Tensor]]:
    for layer in reversed(network.layers):
        if (
            isinstance(layer, ReLU)
            or isinstance(layer, Sigmoid)
            or isinstance(layer, Tanh)
        ):
            assert isinstance(
                layer, ReLU
            ), "active constraint score not supported with sigmoid or tanh layers."
            direct_effect = _compute_active_constraint_score(
                optimizable_parameters,
                subproblem_state,
                batch_size,
                device,
                layer,
            )
            score[layer_tag(layer)] = direct_effect.squeeze(1)
            backup_score[layer_tag(layer)] = -1 * direct_effect.squeeze(1)
        elif isinstance(layer, Sequential):
            score, backup_score = _compute_active_constraint_scores_sequential(
                score,
                backup_score,
                layer,
                optimizable_parameters,
                subproblem_state,
                batch_size,
                device,
            )
        elif isinstance(layer, ResidualBlock):
            score, backup_score = _compute_active_constraint_scores_sequential(
                score,
                backup_score,
                layer.path_a,
                optimizable_parameters,
                subproblem_state,
                batch_size,
                device,
            )
            score, backup_score = _compute_active_constraint_scores_sequential(
                score,
                backup_score,
                layer.path_b,
                optimizable_parameters,
                subproblem_state,
                batch_size,
                device,
            )
    return score, backup_score


@torch.no_grad()
def _compute_split_scores(
    backsubstitution_config: BacksubstitutionConfig,
    query_coef: Tensor,
    network: AbstractNetwork,
    subproblem_state: SubproblemState,  # parameter get modified, so this can not be made readonly
    batch_size: int,
    device: torch.device,
    use_optimized_slopes: bool,
    use_beta_contributions: bool,
    propagation_effect_mode: PropagationEffectMode,
    use_indirect_effect: bool,
    lower_bound_reduce_op: Callable[[Tensor, Tensor], Tensor],
    use_abs: bool,
) -> Tuple[Dict[LayerTag, Tensor], Dict[LayerTag, Tensor]]:
    batch_repeats = batch_size, *([1] * (len(query_coef.shape) - 1))
    batch_query_coef = query_coef.repeat(batch_repeats).to(device)

    abstract_shape = MN_BaB_Shape(
        query_id=query_tag(network),
        query_prev_layer=None,  # not tracked
        queries_to_compute=None,  # not tracked
        lb=AffineForm(batch_query_coef),
        ub=AffineForm(batch_query_coef),
        unstable_queries=None,  # not tracked
        subproblem_state=subproblem_state,
    )
    score: Dict[LayerTag, Tensor] = {}
    contribution_fractions: Dict[LayerTag, Dict[LayerTag, Tensor]] = {}
    backup_score: Dict[LayerTag, Tensor] = {}

    network.reset_input_bounds()
    network.set_intermediate_input_bounds(
        subproblem_state.constraints.layer_bounds.intermediate_bounds
    )  # TODO: get rid of this method
    if not use_optimized_slopes:
        abstract_shape.change_alphas_to_WK_slopes()
    if not use_beta_contributions:
        abstract_shape.set_beta_parameters_to_zero()

    score, backup_score, __ = _compute_split_scores_sequential(
        backsubstitution_config,
        abstract_shape,
        network,
        score,
        backup_score,
        contribution_fractions,
        propagation_effect_mode,
        use_indirect_effect,
        lower_bound_reduce_op,
        use_abs,
    )
    return score, backup_score


def _compute_split_scores_sequential(  # noqa C901
    backsubstitution_config: BacksubstitutionConfig,
    abstract_shape: MN_BaB_Shape,
    network: Sequential,
    score: Dict[LayerTag, Tensor],
    backup_score: Dict[LayerTag, Tensor],
    contribution_fractions: Dict[LayerTag, Dict[LayerTag, Tensor]],
    propagation_effect_mode: PropagationEffectMode,
    use_indirect_effect: bool,
    lower_bound_reduce_op: Callable[[Tensor, Tensor], Tensor],
    use_abs: bool,
) -> Tuple[
    Dict[LayerTag, Tensor],
    Dict[LayerTag, Tensor],
    Dict[LayerTag, Dict[LayerTag, Tensor]],
]:
    assert abstract_shape.subproblem_state is not None
    assert abstract_shape.subproblem_state.constraints.split_state is not None
    for layer_idx, layer in reversed(list(enumerate(network.layers))):
        # NOTE @Robin Custom version for Sigmoid
        if (
            isinstance(layer, ReLU)
            or isinstance(layer, Sigmoid)
            or isinstance(layer, Tanh)
        ):
            previous_layer = network.layers[layer_idx - 1]
            current_layer_lower_bounds = abstract_shape.subproblem_state.constraints.layer_bounds.intermediate_bounds[
                layer_tag(layer)
            ][
                0
            ]
            current_layer_upper_bounds = abstract_shape.subproblem_state.constraints.layer_bounds.intermediate_bounds[
                layer_tag(layer)
            ][
                1
            ]
            assert isinstance(layer, ReLU)  # TODO: get rid of this
            (
                direct_effect,
                propagation_effect,
            ) = _compute_direct_and_propagation_effect_on_lower_bound(
                abstract_shape,
                layer,
                previous_layer,
                current_layer_lower_bounds,
                current_layer_upper_bounds,
                propagation_effect_mode,
                lower_bound_reduce_op,
            )
            current_layer_score = direct_effect + propagation_effect
            if use_indirect_effect:
                contribution_fractions_to_current_layer = (
                    _compute_contribution_fractions_to_layer_bounds(
                        backsubstitution_config,
                        network,
                        query_tag(layer),
                        layer_idx,
                        abstract_shape.subproblem_state,
                        abstract_shape.batch_size,
                        propagation_effect_mode,
                        lower_bound_reduce_op,
                    )
                )
                for (
                    contributing_layer_id,
                    fractions,
                ) in contribution_fractions_to_current_layer.items():
                    if contributing_layer_id not in contribution_fractions:
                        contribution_fractions[contributing_layer_id] = {}
                    contribution_fractions[contributing_layer_id][
                        layer_tag(layer)
                    ] = fractions

                indirect_effect = _compute_indirect_effect(
                    contribution_fractions,
                    score,
                    layer_tag(layer),
                    current_layer_score.shape,
                    current_layer_score.device,
                )
                current_layer_score += indirect_effect
            if use_abs:
                current_layer_score = abs(current_layer_score)
            score[layer_tag(layer)] = current_layer_score.squeeze(1)
            backup_score[layer_tag(layer)] = -1 * direct_effect.squeeze(1)
            abstract_shape = layer.backsubstitute(
                backsubstitution_config, abstract_shape
            )
        elif isinstance(layer, Sequential):
            (
                score,
                backup_score,
                contribution_fractions,
            ) = _compute_split_scores_sequential(
                backsubstitution_config,
                abstract_shape,
                layer,
                score,
                backup_score,
                contribution_fractions,
                propagation_effect_mode,
                use_indirect_effect,
                lower_bound_reduce_op,
                use_abs,
            )

        elif isinstance(layer, ResidualBlock):
            in_lb = abstract_shape.lb.clone()
            assert abstract_shape.ub is not None
            in_ub = abstract_shape.ub.clone()

            (
                score,
                backup_score,
                contribution_fractions,
            ) = _compute_split_scores_sequential(
                backsubstitution_config,
                abstract_shape,
                layer.path_a,
                score,
                backup_score,
                contribution_fractions,
                propagation_effect_mode,
                use_indirect_effect,
                lower_bound_reduce_op,
                use_abs,
            )

            a_lb = abstract_shape.lb.clone()
            a_ub = abstract_shape.ub.clone()

            abstract_shape.update_bounds(in_lb, in_ub)
            (
                score,
                backup_score,
                contribution_fractions,
            ) = _compute_split_scores_sequential(
                backsubstitution_config,
                abstract_shape,
                layer.path_b,
                score,
                backup_score,
                contribution_fractions,
                propagation_effect_mode,
                use_indirect_effect,
                lower_bound_reduce_op,
                use_abs,
            )

            new_lb_bias = (
                a_lb.bias + abstract_shape.lb.bias - in_lb.bias
            )  # Both the shape in a and in b  contain the initial bias terms, so one has to be subtracted
            new_ub_bias = a_ub.bias + abstract_shape.ub.bias - in_ub.bias
            new_lb_coef = a_lb.coef + abstract_shape.lb.coef
            new_ub_coef = a_ub.coef + abstract_shape.ub.coef

            abstract_shape.update_bounds(
                AffineForm(new_lb_coef, new_lb_bias),
                AffineForm(new_ub_coef, new_ub_bias),
            )  # TODO look at merging of dependence sets

        elif isinstance(layer, MultiPathBlock):
            in_lb = abstract_shape.lb.clone()
            assert abstract_shape.ub is not None
            in_ub = abstract_shape.ub.clone()

            pre_merge_shapes = layer.merge.backsubstitute(
                backsubstitution_config, abstract_shape
            )
            post_path_shapes: List[MN_BaB_Shape] = []
            for path_shape, path in zip(pre_merge_shapes, layer.paths):
                (
                    score,
                    backup_score,
                    contribution_fractions,
                ) = _compute_split_scores_sequential(
                    backsubstitution_config,
                    path_shape,
                    path,
                    score,
                    backup_score,
                    contribution_fractions,
                    propagation_effect_mode,
                    use_indirect_effect,
                    lower_bound_reduce_op,
                    use_abs,
                )
                post_path_shapes.append(path_shape)

            if layer.header is not None:
                post_header_shape = layer.header.backsubstitute(
                    backsubstitution_config, post_path_shapes
                )
            else:  # All paths are from the same input we can add them up
                final_lb_form = post_path_shapes[0].lb
                final_ub_form: Optional[AffineForm] = None
                if post_path_shapes[0].ub is not None:
                    final_ub_form = post_path_shapes[0].ub

                for abs_shape in post_path_shapes[1:]:
                    final_lb_form.coef += abs_shape.lb.coef
                    final_lb_form.bias += abs_shape.lb.bias

                    if abs_shape.ub is not None:
                        assert final_ub_form is not None
                        final_ub_form.coef += abs_shape.ub.coef
                        final_ub_form.bias += abs_shape.ub.bias

                post_header_shape = abstract_shape.clone_with_new_bounds(
                    final_lb_form, final_ub_form
                )

            # Adjust bias
            new_lower: AffineForm
            new_upper: Optional[AffineForm] = None

            new_lb_bias = (
                post_header_shape.lb.bias - (len(layer.paths) - 1) * in_lb.bias
            )  # Both the shape in a and in b  contain the initial bias terms, so one has to be subtracted
            new_lb_coef = post_header_shape.lb.coef

            new_lower = AffineForm(new_lb_coef, new_lb_bias)

            if post_header_shape.ub is not None and in_ub is not None:
                new_ub_bias = (
                    post_header_shape.ub.bias - (len(layer.paths) - 1) * in_ub.bias
                )
                new_ub_coef = post_header_shape.ub.coef
                new_upper = AffineForm(new_ub_coef, new_ub_bias)

            abstract_shape.update_bounds(new_lower, new_upper)

        elif isinstance(layer, SplitBlock):
            # pass
            coef_split_dim = layer.split_dim + 2

            # Get the output bounds of the center path
            # center_path_out_lb, abstract_shape.subproblem_state.constraints.layer_bounds
            assert abstract_shape.subproblem_state is not None
            (
                center_path_out_lb,
                center_path_out_ub,
            ) = abstract_shape.subproblem_state.constraints.layer_bounds.intermediate_bounds[
                layer_tag(layer.abs_center_path.layers[-1])
            ]
            assert isinstance(center_path_out_lb, Tensor)
            assert isinstance(center_path_out_ub, Tensor)
            center_path_out_lb = F.relu(center_path_out_lb)
            center_path_out_ub = F.relu(center_path_out_ub)

            assert (center_path_out_lb <= center_path_out_ub + 1e-10).all()
            # Get the lower and upper-bound slopes and offsets for the multiplication
            assert layer.res_lower is not None
            assert layer.res_upper is not None
            res_lower = layer.res_lower
            res_upper = layer.res_upper
            mul_factors = (res_lower, res_upper)
            mul_convex_bounds = layer._get_multiplication_slopes_and_intercepts(
                mul_factors, (center_path_out_lb, center_path_out_ub)
            )

            # Get the input bounds for the dividend

            (
                div_input_lb_lb,
                div_input_lb_ub,
                div_input_ub_lb,
                div_input_ub_ub,
            ) = layer._get_mul_lbs_and_ubs(
                mul_factors, (center_path_out_lb, center_path_out_ub)
            )
            div_input_lb = torch.minimum(div_input_lb_lb, div_input_ub_lb).sum(
                dim=layer.outer_reduce_dim + 1
            )
            div_input_ub = torch.maximum(div_input_lb_ub, div_input_ub_ub).sum(
                dim=layer.outer_reduce_dim + 1
            )
            div_input_bounds = (
                div_input_lb,
                div_input_ub,
            )

            # Get the lower and upper-bound slopes and offsets for the division
            div_factors = (
                1 / res_lower.sum(dim=layer.outer_reduce_dim + 1),
                1 / res_upper.sum(dim=layer.outer_reduce_dim + 1),
            )
            div_convex_bounds = layer._get_multiplication_slopes_and_intercepts(
                div_factors, div_input_bounds
            )

            # Backpropagation Part 1 Div-Reshape
            lower_form = layer._backsub_affine_form_first(
                abstract_shape.lb, div_convex_bounds, False, abstract_shape
            )
            upper_form: Optional[AffineForm] = None
            if abstract_shape.ub is not None:
                upper_form = layer._backsub_affine_form_first(
                    abstract_shape.ub, div_convex_bounds, True, abstract_shape
                )

            # Update Abstract Shape so that we can go through mul layer
            abstract_shape.update_bounds(lower_form, upper_form)

            # Backprop Part 2 - Mul
            lower_form = layer._backsub_affine_form_given_convex_bounds(
                abstract_shape.lb, mul_convex_bounds, False, abstract_shape
            )
            if abstract_shape.ub is not None:
                upper_form = layer._backsub_affine_form_given_convex_bounds(
                    abstract_shape.ub, mul_convex_bounds, True, abstract_shape
                )

            # Update Abstract Shape
            abstract_shape.update_bounds(lower_form, upper_form)

            # Backprop center_path
            (
                score,
                backup_score,
                contribution_fractions,
            ) = _compute_split_scores_sequential(
                backsubstitution_config,
                abstract_shape,
                layer.abs_center_path,
                score,
                backup_score,
                contribution_fractions,
                propagation_effect_mode,
                use_indirect_effect,
                lower_bound_reduce_op,
                use_abs,
            )

            # Backprop through the split
            # As we concretized the second split, we simply append it with 0 sensitivity
            # NOTE: Not generalized for arbitrary splits (assumes only 2 splits)
            assert len(layer.split[1]) == 2
            assert isinstance(abstract_shape.lb.coef, Tensor)
            zero_append_shape = [
                abstract_shape.lb.coef.shape[0],
                abstract_shape.lb.coef.shape[1],
                *layer.input_dim,
            ]
            zero_append_shape[coef_split_dim] = layer.split[1][1]

            zero_append_matrix = torch.zeros(
                zero_append_shape, device=abstract_shape.device
            )
            zero_appended_lb = torch.cat(
                (abstract_shape.lb.coef, zero_append_matrix), dim=coef_split_dim
            )

            lower_form = AffineForm(zero_appended_lb, abstract_shape.lb.bias)

            if abstract_shape.ub is not None:
                assert isinstance(abstract_shape.ub.coef, Tensor)
                zero_appended_ub = torch.cat(
                    (abstract_shape.ub.coef, zero_append_matrix), dim=coef_split_dim
                )
                upper_form = AffineForm(zero_appended_ub, abstract_shape.ub.bias)

            abstract_shape.update_bounds(lower_form, upper_form)

        else:
            abstract_shape = layer.backsubstitute(
                backsubstitution_config, abstract_shape
            )

    return score, backup_score, contribution_fractions


def babsr_ratio_computation(
    lower_bound: Tensor, upper_bound: Tensor
) -> Tuple[Tensor, Tensor]:
    lower_temp = lower_bound - F.relu(lower_bound)
    upper_temp = F.relu(upper_bound)
    slope_ratio = upper_temp / (upper_temp - lower_temp)
    intercept = -1 * lower_temp * slope_ratio

    return slope_ratio.nan_to_num(), intercept.nan_to_num()


def _compute_active_constraint_score(
    optimizable_parameters: ReadonlyParametersForQuery,
    subproblem_state: ReadonlySubproblemState,
    batch_size: int,
    device: torch.device,
    layer: ReLU,
) -> Tensor:
    assert subproblem_state.constraints.split_state is not None
    split_state = subproblem_state.constraints.split_state
    assert subproblem_state.constraints.prima_constraints is not None
    prima_constraints = subproblem_state.constraints.prima_constraints
    if isinstance(layer, ReLU):  # TODO: move this logic entirely into SplitState?
        unstable_nodes_mask = split_state.unstable_node_mask_in_layer(
            layer_tag(layer)
        ).unsqueeze(1)
    else:
        unstable_nodes_mask = torch.ones_like(
            split_state.split_constraints[layer_tag(layer)].unsqueeze(1),
            dtype=torch.bool,
            device=split_state.device,  # TODO: can we just use "device" here?
        )
    if (
        prima_constraints is None
        or layer_tag(layer) not in prima_constraints.prima_coefficients
        or prima_constraints.prima_coefficients[layer_tag(layer)][0].shape[2] == 0
    ):
        return torch.zeros(batch_size, 1, *layer.output_dim, device=device)

    (
        current_layer_prima_output_coefficients,
        current_layer_prima_input_coefficients,
        __,
    ) = prima_constraints.prima_coefficients[layer_tag(layer)]
    prima_parameters = optimizable_parameters.parameters[key_prima_lb][layer_tag(layer)]
    prima_output_contribution = layer._multiply_prima_coefs_and_parameters(
        torch.sqrt(
            torch.square(current_layer_prima_output_coefficients)
        ),  # abs not available for sparse tensors
        prima_parameters,
    )
    prima_input_contribution = layer._multiply_prima_coefs_and_parameters(
        torch.sqrt(
            torch.square(current_layer_prima_input_coefficients)
        ),  # abs not available for sparse tensors
        prima_parameters,
    )
    prima_contribution = prima_input_contribution + prima_output_contribution

    return (prima_contribution) * unstable_nodes_mask


def _compute_direct_and_propagation_effect_on_lower_bound(
    abstract_shape: MN_BaB_Shape,
    layer: ReLU,
    previous_layer: Union[Linear, Conv2d],
    current_layer_lower_bounds: Tensor,
    current_layer_upper_bounds: Tensor,
    propagation_effect_mode: PropagationEffectMode,
    lower_bound_reduce_op: Callable[[Tensor, Tensor], Tensor],
) -> Tuple[Tensor, Tensor]:
    assert abstract_shape.subproblem_state is not None
    subproblem_state = abstract_shape.subproblem_state
    assert subproblem_state.constraints.split_state is not None
    if (  # TODO: move this logic into PrimaConstraints? ( seems to be duplicated)
        subproblem_state.constraints.prima_constraints
        and layer_tag(layer)
        in subproblem_state.constraints.prima_constraints.prima_coefficients
        and subproblem_state.constraints.prima_constraints.prima_coefficients[
            layer_tag(layer)
        ][0].shape[-1]
        > 0
    ):
        (
            current_layer_prima_output_coefficients,
            current_layer_prima_input_coefficients,
            __,
        ) = subproblem_state.constraints.prima_constraints.prima_coefficients[
            layer_tag(layer)
        ]
        prima_parameters = abstract_shape.get_existing_parameters(
            key_prima_lb, layer_tag(layer)
        )
        prima_output_contribution = layer._multiply_prima_coefs_and_parameters(
            current_layer_prima_output_coefficients, prima_parameters
        )
        prima_input_contribution = layer._multiply_prima_coefs_and_parameters(
            current_layer_prima_input_coefficients, prima_parameters
        )
    else:
        assert not abstract_shape.uses_dependence_sets()
        assert isinstance(abstract_shape.lb.coef, Tensor)
        prima_output_contribution = torch.zeros_like(abstract_shape.lb.coef)
        prima_input_contribution = torch.zeros_like(abstract_shape.lb.coef)

    lb_coef_before_relaxation = abstract_shape.lb.coef + prima_output_contribution

    lb_slope = abstract_shape.get_existing_parameters(
        key_alpha_relu_lb, layer_tag(layer)
    )
    ub_slope, ub_intercept = babsr_ratio_computation(
        current_layer_lower_bounds, current_layer_upper_bounds
    )

    (
        neg_lb_coef_before_relaxation,
        pos_lb_coef_before_relaxation,
    ) = get_neg_pos_comp(lb_coef_before_relaxation)

    beta_parameters = abstract_shape.get_existing_parameters(
        key_beta_lb, layer_tag(layer)
    )
    beta_contribution_shape = (abstract_shape.batch_size, 1, *layer.output_dim)
    beta_contribution = (
        beta_parameters
        * subproblem_state.constraints.split_state.split_constraints[layer_tag(layer)]
    ).view(beta_contribution_shape)
    lb_coef = (
        (
            pos_lb_coef_before_relaxation * lb_slope
            + neg_lb_coef_before_relaxation * ub_slope
        )
        + prima_input_contribution
        + beta_contribution
    )
    neg_lb_coef, pos_lb_coef = get_neg_pos_comp(lb_coef)

    previous_layer_bias = _get_layer_bias(previous_layer, lb_coef.dim())

    (
        negative_coef_multiplier_before,
        positive_coef_multiplier_before,
        negative_coef_multiplier_neg_split,
        positive_coef_multiplier_neg_split,
        negative_coef_multiplier_pos_split,
        positive_coef_multiplier_pos_split,
    ) = _get_coef_multipliers(
        current_layer_lower_bounds,
        current_layer_upper_bounds,
        previous_layer_bias,
        propagation_effect_mode,
    )

    propagation_contribution_before = (
        neg_lb_coef * negative_coef_multiplier_before
        + pos_lb_coef * positive_coef_multiplier_before
    )
    neg_lb_coef_neg_split, pos_lb_coef_neg_split = get_neg_pos_comp(
        prima_input_contribution + beta_contribution
    )
    propagation_contribution_neg_split = (
        neg_lb_coef_neg_split * negative_coef_multiplier_neg_split
        + pos_lb_coef_neg_split * positive_coef_multiplier_neg_split
    )
    neg_lb_coef_pos_split, pos_lb_coef_pos_split = get_neg_pos_comp(
        lb_coef_before_relaxation + prima_input_contribution + beta_contribution
    )
    propagation_contribution_pos_split = (
        neg_lb_coef_pos_split * negative_coef_multiplier_pos_split
        + pos_lb_coef_pos_split * positive_coef_multiplier_pos_split
    )
    propagation_effect_neg_split = (
        propagation_contribution_neg_split - propagation_contribution_before
    )
    propagation_effect_pos_split = (
        propagation_contribution_pos_split - propagation_contribution_before
    )
    unstable_nodes_mask = (
        subproblem_state.constraints.split_state.split_constraints[layer_tag(layer)]
        == 0
    ).unsqueeze(1)
    propagation_effect = lower_bound_reduce_op(
        propagation_effect_neg_split, propagation_effect_pos_split
    ) * (unstable_nodes_mask)

    direct_effect = -1 * (
        neg_lb_coef_before_relaxation * ub_intercept * unstable_nodes_mask
    )
    assert (direct_effect >= 0).all()
    return direct_effect, propagation_effect


def _get_layer_bias(previous_layer: AbstractModule, coef_dim: int) -> Tensor:
    if isinstance(previous_layer, Sequential):
        previous_layer_bias = previous_layer.get_babsr_bias()
    else:
        assert isinstance(previous_layer.bias, Tensor)
        previous_layer_bias = previous_layer.bias
    # unsqueeze to batch_dim, query_dim, bias_dim
    previous_layer_bias = previous_layer_bias.unsqueeze(0).unsqueeze(0)
    expected_number_of_coef_dims_if_prev_layer_is_conv = 5
    if coef_dim == expected_number_of_coef_dims_if_prev_layer_is_conv:
        # unsqueeze bias from batch_dim, query_sim, channel to batch_dim, query_sim, channel, height, width
        previous_layer_bias = previous_layer_bias.unsqueeze(-1).unsqueeze(-1)
    # assert coef_dim == previous_layer_bias.dim(), "bias expanded to unexpected shape"
    return previous_layer_bias


def _compute_direct_and_propagation_effect_on_upper_bound(
    abstract_shape: MN_BaB_Shape,
    layer: ReLU,
    previous_layer: Union[Linear, Conv2d],
    current_layer_lower_bounds: Tensor,
    current_layer_upper_bounds: Tensor,
    propagation_effect_mode: PropagationEffectMode,
    lower_bound_reduce_op: Callable[[Tensor, Tensor], Tensor],
) -> Tuple[Tensor, Tensor]:
    assert abstract_shape.subproblem_state is not None
    subproblem_state = abstract_shape.subproblem_state
    assert subproblem_state.constraints.split_state is not None
    assert abstract_shape.ub is not None
    if (  # TODO: move this logic into PrimaConstraints (seems to be duplicated)
        subproblem_state.constraints.prima_constraints
        and layer_tag(layer)
        in subproblem_state.constraints.prima_constraints.prima_coefficients
        and subproblem_state.constraints.prima_constraints.prima_coefficients[
            layer_tag(layer)
        ][0].shape[-1]
        > 0
    ):
        (
            current_layer_prima_output_coefficients,
            current_layer_prima_input_coefficients,
            __,
        ) = subproblem_state.constraints.prima_constraints.prima_coefficients[
            layer_tag(layer)
        ]
        prima_parameters = abstract_shape.get_existing_parameters(
            key_prima_ub, layer_tag(layer)
        )
        prima_output_contribution = layer._multiply_prima_coefs_and_parameters(
            current_layer_prima_output_coefficients, prima_parameters
        )
        prima_input_contribution = layer._multiply_prima_coefs_and_parameters(
            current_layer_prima_input_coefficients, prima_parameters
        )
    else:
        assert not abstract_shape.uses_dependence_sets()
        assert isinstance(abstract_shape.ub.coef, Tensor)
        prima_output_contribution = torch.zeros_like(abstract_shape.ub.coef)
        prima_input_contribution = torch.zeros_like(abstract_shape.ub.coef)

    ub_coef_before_relaxation = abstract_shape.ub.coef - prima_output_contribution

    lb_slope = abstract_shape.get_existing_parameters(
        key_alpha_relu_ub, layer_tag(layer)
    )  # TODO: should this be key_alpha_relu_ub?
    ub_slope, ub_intercept = babsr_ratio_computation(
        current_layer_lower_bounds, current_layer_upper_bounds
    )

    (
        neg_ub_coef_before_relaxation,
        pos_ub_coef_before_relaxation,
    ) = get_neg_pos_comp(ub_coef_before_relaxation)

    beta_parameters = abstract_shape.get_existing_parameters(
        key_beta_ub, layer_tag(layer)
    )
    beta_contribution_shape = (abstract_shape.batch_size, 1, *layer.output_dim)
    beta_contribution = (
        beta_parameters
        * subproblem_state.constraints.split_state.split_constraints[layer_tag(layer)]
    ).view(beta_contribution_shape)
    ub_coef = (
        (
            pos_ub_coef_before_relaxation * ub_slope
            + neg_ub_coef_before_relaxation * lb_slope
        )
        - prima_input_contribution
        - beta_contribution
    )
    neg_ub_coef, pos_ub_coef = get_neg_pos_comp(ub_coef)

    previous_layer_bias = _get_layer_bias(previous_layer, ub_coef.dim())

    (
        negative_coef_multiplier_before,
        positive_coef_multiplier_before,
        negative_coef_multiplier_neg_split,
        positive_coef_multiplier_neg_split,
        negative_coef_multiplier_pos_split,
        positive_coef_multiplier_pos_split,
    ) = _get_coef_multipliers(
        current_layer_lower_bounds,
        current_layer_upper_bounds,
        previous_layer_bias,
        propagation_effect_mode,
        for_lower_bound=False,
    )

    propagation_contribution_before = (
        neg_ub_coef * negative_coef_multiplier_before
        + pos_ub_coef * positive_coef_multiplier_before
    )
    neg_ub_coef_neg_split, pos_ub_coef_neg_split = get_neg_pos_comp(
        -prima_input_contribution - beta_contribution
    )
    propagation_contribution_neg_split = (
        neg_ub_coef_neg_split * negative_coef_multiplier_neg_split
        + pos_ub_coef_neg_split * positive_coef_multiplier_neg_split
    )
    neg_ub_coef_pos_split, pos_ub_coef_pos_split = get_neg_pos_comp(
        ub_coef_before_relaxation - prima_input_contribution - beta_contribution
    )
    propagation_contribution_pos_split = (
        neg_ub_coef_pos_split * negative_coef_multiplier_pos_split
        + pos_ub_coef_pos_split * positive_coef_multiplier_pos_split
    )
    propagation_effect_neg_split = (
        propagation_contribution_neg_split - propagation_contribution_before
    )
    propagation_effect_pos_split = (
        propagation_contribution_pos_split - propagation_contribution_before
    )
    unstable_nodes_mask = (
        subproblem_state.constraints.split_state.split_constraints[layer_tag(layer)]
        == 0
    ).unsqueeze(1)
    propagation_effect = lower_bound_reduce_op(
        propagation_effect_neg_split, propagation_effect_pos_split
    ) * (unstable_nodes_mask)

    direct_effect = -1 * (
        pos_ub_coef_before_relaxation * ub_intercept * unstable_nodes_mask
    )
    assert (direct_effect <= 0).all()
    return direct_effect, propagation_effect


def _get_coef_multipliers(
    layer_lower_bounds: Tensor,
    layer_upper_bounds: Tensor,
    prev_layer_bias: Tensor,
    propagation_effect_mode: PropagationEffectMode,
    for_lower_bound: bool = True,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    zero = torch.zeros_like(layer_lower_bounds)
    if propagation_effect_mode == PropagationEffectMode.none:
        return zero, zero, zero, zero, zero, zero
    elif propagation_effect_mode == PropagationEffectMode.bias:
        return (
            prev_layer_bias,
            prev_layer_bias,
            prev_layer_bias,
            prev_layer_bias,
            prev_layer_bias,
            prev_layer_bias,
        )
    elif propagation_effect_mode == PropagationEffectMode.intermediate_concretization:
        negative_coef_multiplier_before = layer_upper_bounds
        positive_coef_multiplier_before = layer_lower_bounds
        negative_coef_multiplier_neg_split = zero
        positive_coef_multiplier_neg_split = layer_lower_bounds
        negative_coef_multiplier_pos_split = layer_upper_bounds
        positive_coef_multiplier_pos_split = zero

        if for_lower_bound:
            return (
                negative_coef_multiplier_before,
                positive_coef_multiplier_before,
                negative_coef_multiplier_neg_split,
                positive_coef_multiplier_neg_split,
                negative_coef_multiplier_pos_split,
                positive_coef_multiplier_pos_split,
            )
        else:
            return (
                positive_coef_multiplier_before,
                negative_coef_multiplier_before,
                positive_coef_multiplier_neg_split,
                negative_coef_multiplier_neg_split,
                positive_coef_multiplier_pos_split,
                negative_coef_multiplier_pos_split,
            )
    else:
        raise RuntimeError(
            'Unexpected propagation effect mode option, allowed options are "none", "bias" and "intermediate_concretization".'
        )


def _compute_contribution_fractions_to_layer_bounds(
    backsubstitution_config: BacksubstitutionConfig,
    network: Sequential,
    query_id: QueryTag,
    starting_layer_index: int,
    subproblem_state: SubproblemState,  # TODO: can this be made readonly?
    batch_size: int,
    propagation_effect_mode: PropagationEffectMode,
    lower_bound_reduce_op: Callable[[Tensor, Tensor], Tensor],
) -> Dict[LayerTag, Tensor]:
    assert query_tag(network.layers[starting_layer_index]) == query_id
    assert isinstance(network.layers[starting_layer_index], ReLU)
    layer_shape = network.layers[starting_layer_index - 1].output_dim
    device = subproblem_state.device

    intermediate_bounds_to_recompute = None  # (full set of queries)

    query_coef = get_output_bound_initial_query_coef(
        dim=layer_shape,
        batch_size=batch_size,
        intermediate_bounds_to_recompute=intermediate_bounds_to_recompute,
        use_dependence_sets=False,  # TODO: why False? (does it even matter for branching score computations?)
        device=device,
        dtype=None,  # TODO: should this be something else?
    )

    # subproblem_state_for_bounds=subproblem_state
    # if use_dependence_sets_for_current_bounds:
    #     subproblem_state_for_bounds=subproblem_state.without_prima() # TODO: get rid of this?

    abstract_shape = MN_BaB_Shape(
        query_id=query_id,
        query_prev_layer=None,  # (not tracked)
        queries_to_compute=intermediate_bounds_to_recompute,
        lb=AffineForm(query_coef),
        ub=AffineForm(query_coef),
        unstable_queries=None,  # (not tracked)
        subproblem_state=subproblem_state,
    )

    contribution_fraction_to_starting_layer = {}

    starting_layer_lower_bounds = (
        subproblem_state.constraints.layer_bounds.intermediate_bounds[
            layer_from_query_tag(query_id)
        ][0]
    )
    starting_layer_upper_bounds = (
        subproblem_state.constraints.layer_bounds.intermediate_bounds[
            layer_from_query_tag(query_id)
        ][1]
    )
    assert abstract_shape.subproblem_state is subproblem_state
    assert subproblem_state.constraints.split_state is not None
    for layer_idx, layer in reversed(
        list(enumerate(network.layers[:starting_layer_index]))
    ):
        if isinstance(layer, ReLU):
            current_layer_lower_bounds = (
                subproblem_state.constraints.layer_bounds.intermediate_bounds[
                    layer_tag(layer)
                ][0]
            )
            current_layer_upper_bounds = (
                subproblem_state.constraints.layer_bounds.intermediate_bounds[
                    layer_tag(layer)
                ][1]
            )
            assert abstract_shape.ub is not None
            assert isinstance(abstract_shape.ub.coef, Tensor)
            starting_and_affected_node_unstable_mask = (
                subproblem_state.constraints.split_state.split_constraints[
                    layer_tag(layer)
                ]
                == 0
            ).unsqueeze(1) * _reshape_layer_values(
                subproblem_state.constraints.split_state.split_constraints[
                    layer_from_query_tag(query_id)
                ]
                == 0,
                len(abstract_shape.ub.coef.shape),
            )

            previous_layer = network.layers[layer_idx - 1]
            (
                lb_direct_effect,
                lb_propagation_effect,
            ) = _compute_direct_and_propagation_effect_on_lower_bound(
                abstract_shape,
                layer,
                previous_layer,
                current_layer_lower_bounds,
                current_layer_upper_bounds,
                propagation_effect_mode,
                lower_bound_reduce_op,
            )
            lb_contribution = (
                -1
                * (lb_direct_effect + lb_propagation_effect)
                * starting_and_affected_node_unstable_mask
            )
            (
                ub_direct_effect,
                ub_propagation_effect,
            ) = _compute_direct_and_propagation_effect_on_upper_bound(
                abstract_shape,
                layer,
                previous_layer,
                current_layer_lower_bounds,
                current_layer_upper_bounds,
                propagation_effect_mode,
                _get_opposite_operation(lower_bound_reduce_op),
            )
            ub_contribution = (
                -1
                * (ub_direct_effect + ub_propagation_effect)
                * starting_and_affected_node_unstable_mask
            )
            assert lb_contribution.shape == ub_contribution.shape
            assert lb_contribution.shape[1] == np.prod(layer_shape)
            assert lb_contribution.shape[2:] == layer.output_dim

            contribution_fraction_to_starting_layer[
                layer_tag(layer)
            ] = _compute_triangle_relaxation_area_change(
                starting_layer_lower_bounds,
                starting_layer_upper_bounds,
                lb_contribution,
                ub_contribution,
            )

        abstract_shape = layer.backsubstitute(backsubstitution_config, abstract_shape)
    return contribution_fraction_to_starting_layer


def _reshape_layer_values(x: Tensor, number_of_dimensions_to_reshape_to: int) -> Tensor:
    if number_of_dimensions_to_reshape_to == 5:
        return x.flatten(start_dim=1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    elif number_of_dimensions_to_reshape_to == 3:
        return x.flatten(start_dim=1).unsqueeze(-1)
    else:
        raise RuntimeError("Unexpected number of dimensions encountered.")


def _get_opposite_operation(
    op: Callable[[Tensor, Tensor], Tensor]
) -> Callable[[Tensor, Tensor], Tensor]:
    if op == torch.maximum:
        return torch.minimum
    elif op == torch.minimum:
        return torch.maximum
    else:
        raise RuntimeError("Unknown reduce operation for branching")


def _compute_triangle_relaxation_area_change(
    layer_lower_bounds: Tensor,
    layer_upper_bounds: Tensor,
    lower_bound_contribution: Tensor,
    upper_bound_contribution: Tensor,
) -> Tensor:
    unstable_nodes_mask = _reshape_layer_values(
        (layer_lower_bounds < 0) & (layer_upper_bounds > 0),
        len(lower_bound_contribution.shape),
    )
    lb_contribution_fraction = torch.where(
        unstable_nodes_mask,
        lower_bound_contribution
        / _reshape_layer_values(
            layer_lower_bounds, len(lower_bound_contribution.shape)
        ),
        torch.zeros_like(lower_bound_contribution),
    )
    assert (lb_contribution_fraction >= 0).all()
    lb_contribution_fraction = lb_contribution_fraction.clamp(max=1)
    ub_contribution_fraction = torch.where(
        unstable_nodes_mask,
        upper_bound_contribution
        / _reshape_layer_values(
            layer_upper_bounds, len(upper_bound_contribution.shape)
        ),
        torch.zeros_like(lower_bound_contribution),
    )
    assert (ub_contribution_fraction >= 0).all()
    ub_contribution_fraction = ub_contribution_fraction.clamp(max=1)

    contribution_to_triangle_relaxation_area = (
        lb_contribution_fraction
        + ub_contribution_fraction
        - lb_contribution_fraction * ub_contribution_fraction
    )
    assert (contribution_to_triangle_relaxation_area >= 0).all()
    assert (contribution_to_triangle_relaxation_area <= 1).all()
    return contribution_to_triangle_relaxation_area


def _compute_indirect_effect(
    contribution_fractions: Dict[LayerTag, Dict[LayerTag, Tensor]],
    score: Dict[LayerTag, Tensor],
    current_layer_id: LayerTag,
    expected_shape: Tuple[int, ...],
    device: torch.device,
) -> Tensor:
    indirect_score = torch.zeros(*expected_shape, device=device)
    if current_layer_id in contribution_fractions:
        current_layer_contribution_fractions = contribution_fractions[current_layer_id]
        for l_id, fractions in current_layer_contribution_fractions.items():
            indirect_score += (
                fractions * _reshape_layer_values(score[l_id], len(fractions.shape))
            ).sum(1)
    return indirect_score


def _find_random_node_to_split(
    split_state: ReadonlySplitState, batch_index: int
) -> NodeTag:
    for layer_id in reversed(list(split_state.split_constraints.keys())):
        try:
            nodes_not_yet_split_in_layer = split_state.split_constraints[layer_id] == 0
            unstable_neuron_indices_in_layer = torch.nonzero(
                nodes_not_yet_split_in_layer[batch_index]
            )
            n_unstable_neurons = unstable_neuron_indices_in_layer.shape[0]
            random_unstable_neuron_index = random.randint(0, n_unstable_neurons - 1)
            random_unstable_neuron = tuple(
                unstable_neuron_indices_in_layer[random_unstable_neuron_index].tolist(),
            )

            break
        except ValueError:
            continue

    return NodeTag(layer=layer_id, index=random_unstable_neuron)


def _adjust_based_on_cost(
    scores: Dict[LayerTag, Tensor],
    backup_scores: Dict[LayerTag, Tensor],
    split_cost_by_layer: Dict[LayerTag, float],
) -> Tuple[Dict[LayerTag, Tensor], Dict[LayerTag, Tensor]]:
    for layer_id, split_cost in split_cost_by_layer.items():
        assert layer_id in scores
        assert layer_id in backup_scores
        scores[layer_id] = scores[layer_id] / split_cost
        backup_scores[layer_id] = backup_scores[layer_id] / split_cost
    return scores, backup_scores


# adapted from: https://github.com/huanzhang12/alpha-beta-CROWN/blob/main/src/branching_heuristics.py
# commit hash: cdbcba0ea346ebd03d552023773829fe6e0822c7
def find_index_to_split_with_filtered_smart_branching(
    subproblem: ReadonlyVerificationSubproblem,
    network: AbstractNetwork,
    backsubstitution_config: BacksubstitutionConfig,
    query_coef: Tensor,
    split_cost_by_layer: Optional[Dict[LayerTag, float]],
    filtered_smart_branching_config: FilteredSmartBranchingConfig,
    input_lb: Tensor,
    input_ub: Tensor,
    batch_sizes: Sequence[int],
    recompute_intermediate_bounds_after_branching: bool,
    optimizer: MNBabOptimizer,
) -> NodeTag:
    assert (
        not subproblem.is_fully_split
    ), "Can't find a node to split for fully split subproblems."
    number_of_preselected_candidates_per_layer = (
        filtered_smart_branching_config.n_candidates
    )
    lower_bound_reduce_op = _get_lower_bound_reduce_op(
        filtered_smart_branching_config.reduce_op
    )

    device = next(network.parameters()).device
    dtype = input_lb.dtype

    subproblem_state = subproblem.subproblem_state.without_prima().deep_copy_to(device)
    assert subproblem_state.constraints.split_state is not None

    batch_sizes_by_layer_id = {  # TODO: get rid of this hack, there must be a better way to get this information
        layer_id: batch_sizes[layer_index]
        for layer_index, layer_id in enumerate(
            subproblem_state.constraints.layer_bounds.intermediate_bounds.keys()
        )
    }

    batch_size = 1
    nodes_not_yet_split_mask = {  # TODO: make this a member function of Constraints?
        layer_id: (
            (
                subproblem_state.constraints.layer_bounds.intermediate_bounds[layer_id][
                    0
                ]
                < 0
            )
            & (
                subproblem_state.constraints.layer_bounds.intermediate_bounds[layer_id][
                    1
                ]
                > 0
            )
            & (layer_split_constraints == 0)
        ).to(dtype)
        for layer_id, layer_split_constraints in subproblem_state.constraints.split_state.split_constraints.items()
    }

    babsr_scores, intercept_tb = _compute_split_scores(
        backsubstitution_config=backsubstitution_config,
        query_coef=query_coef,
        network=network,
        subproblem_state=subproblem_state,
        batch_size=batch_size,
        device=device,
        use_optimized_slopes=False,
        use_beta_contributions=False,
        propagation_effect_mode=PropagationEffectMode.bias,
        use_indirect_effect=False,
        lower_bound_reduce_op=lower_bound_reduce_op,
        use_abs=True,
    )

    decision: List[NodeTag] = []
    for batch_index in range(batch_size):
        babsr_scores_of_batch_element = {
            k: babsr_scores[k][batch_index] for k in babsr_scores.keys()
        }
        intercept_tb_of_batch_element = {
            k: intercept_tb[k][batch_index] for k in intercept_tb.keys()
        }

        all_candidates: Dict[NodeTag, float] = {}
        for i, layer_id in enumerate(babsr_scores_of_batch_element.keys()):
            if (
                babsr_scores_of_batch_element[layer_id].max() <= 1e-4
                and intercept_tb_of_batch_element[layer_id].min() >= -1e-4
            ):
                print("{}th layer has no valid scores".format(i))
                continue

            topk_indices_from_babsr_scores = _get_indices_of_topk(
                babsr_scores_of_batch_element[layer_id],
                number_of_preselected_candidates_per_layer,
                largest=True,
            )
            topk_indices_from_intercept_tb = _get_indices_of_topk(
                intercept_tb_of_batch_element[layer_id],
                number_of_preselected_candidates_per_layer,
                largest=False,
            )
            unique_topk_indices = list(
                set(topk_indices_from_babsr_scores + topk_indices_from_intercept_tb)
            )

            layer_candidate_nodes_to_split = [
                NodeTag(layer=layer_id, index=candidate_index)
                for candidate_index in unique_topk_indices
                if nodes_not_yet_split_mask[layer_id][(batch_index, *candidate_index)]
            ]
            layer_candidate_scores = _compute_candidate_scores_for(
                layer_candidate_nodes_to_split,
                subproblem,
                optimizer,
                query_coef,
                network,
                input_lb,
                input_ub,
                lower_bound_reduce_op,
                batch_sizes_by_layer_id[layer_id],
                recompute_intermediate_bounds_after_branching,
            )
            for node_split, score in zip(
                layer_candidate_nodes_to_split, layer_candidate_scores
            ):
                all_candidates[node_split] = score

        if split_cost_by_layer is not None:
            all_candidates = _adjust_filtered_smart_branching_scores_based_on_cost(
                all_candidates, split_cost_by_layer
            )
        decision.append(max(all_candidates, key=lambda k: all_candidates[k]))

    assert len(decision) == batch_size
    return decision[0]


def _get_indices_of_topk(x: Tensor, k: int, largest: bool) -> List[Tuple[int, ...]]:
    flattenend_indices = torch.topk(x.flatten(), k, largest=largest).indices.cpu()
    indices_by_dimension = np.unravel_index(flattenend_indices, x.shape)
    return [tuple(indices[i] for indices in indices_by_dimension) for i in range(k)]


def _compute_candidate_scores_for(
    candidate_nodes_to_split: Sequence[NodeTag],
    subproblem: ReadonlyVerificationSubproblem,
    optimizer: MNBabOptimizer,
    query_coef: Tensor,
    network: AbstractNetwork,
    input_lb: Tensor,
    input_ub: Tensor,
    lower_bound_reduce_op: Callable[[Tensor, Tensor], Tensor],
    batch_size_for_bounding: int,
    recompute_intermediate_bounds_after_branching: bool,
) -> Sequence[float]:
    device = input_lb.device
    subproblems_to_bound = [
        split.deep_copy_to(device)
        for node_to_split in candidate_nodes_to_split
        for split in subproblem.split(
            node_to_split,
            recompute_intermediate_bounds_after_branching,
            network.layer_id_to_layer[node_to_split.layer],
        )
    ]

    max_queries = 2 * batch_size_for_bounding
    n_scores_to_compute = len(subproblems_to_bound)

    candidate_scores: List[float] = []
    offset = 0
    while offset < n_scores_to_compute:
        subproblem_batch = batch_subproblems(
            subproblems_to_bound[offset : offset + max_queries],
            reuse_single_subproblem=True,
        )

        batch_repeats = min(offset + max_queries, n_scores_to_compute) - offset, *(
            [1] * (len(query_coef.shape) - 1)
        )
        # Unclear mypy behaviour
        (
            lower_bounds,
            __,
            __,
            __,
        ) = optimizer.bound_minimum_with_deep_poly(  # type:ignore [assignment]
            optimizer.backsubstitution_config,
            input_lb,
            input_ub,
            network,
            query_coef.to(device).repeat(batch_repeats),
            subproblem_state=subproblem_batch.subproblem_state,
            ibp_pass=False,
            reset_input_bounds=False,  # TODO check if this is the desried behaviour for FSB
        )
        assert isinstance(lower_bounds, Sequence)
        candidate_scores += _extract_candidate_scores(
            lower_bounds, reduce_op=lower_bound_reduce_op
        )

        offset += max_queries

    return candidate_scores


def _extract_candidate_scores(
    subproblem_lower_bounds: Sequence[float],
    reduce_op: Callable[[Tensor, Tensor], Tensor],
) -> Sequence[float]:
    assert (
        len(subproblem_lower_bounds) % 2 == 0
    ), "Expected an even number of lower bounds."

    lower_bounds_in_pairs = [
        (subproblem_lower_bounds[i], subproblem_lower_bounds[i + 1])
        for i in range(0, len(subproblem_lower_bounds), 2)
    ]

    return [
        reduce_op(torch.tensor(score_pair[0]), torch.tensor(score_pair[1])).item()
        for score_pair in lower_bounds_in_pairs
    ]


def _adjust_filtered_smart_branching_scores_based_on_cost(
    scores: Dict[NodeTag, float], split_cost_by_layer: Dict[LayerTag, float]
) -> Dict[NodeTag, float]:
    for node_to_split, score in scores.items():
        layer_id = node_to_split.layer
        assert layer_id in split_cost_by_layer
        scores[node_to_split] = score / split_cost_by_layer[layer_id]
    return scores


def compute_split_cost_by_layer(
    network: AbstractNetwork,
    prima_coefficients: Dict[LayerTag, Tuple[Tensor, Tensor, Tensor]],
    recompute_intermediate_bounds_after_branching: bool,
) -> Dict[LayerTag, float]:
    if not recompute_intermediate_bounds_after_branching:
        return {layer_id: 1.0 for layer_id in network.get_activation_layer_ids()}
    cost_of_backsubstitution_operation_per_layer = (
        _estimated_cost_of_backsubstitution_operation_per_layer(
            network, prima_coefficients
        )
    )

    (
        cost_of_backsubstitution_pass_starting_at,
        __,
    ) = _estimated_cost_of_backsubstitution_pass_per_layer(
        network, cost_of_backsubstitution_operation_per_layer, 0.0
    )
    cost_of_backsubstitution_pass_starting_at[layer_tag(network)] = (
        sum(cost_of_backsubstitution_operation_per_layer.values()) * 1
    )

    cost_by_layer, __ = _estimated_cost_of_split_at_layer(
        network, cost_of_backsubstitution_pass_starting_at, 0.0
    )

    largest_cost_by_layer = max(cost_by_layer.values())
    for layer_id, costs in cost_by_layer.items():
        cost_by_layer[layer_id] = costs / largest_cost_by_layer
    return cost_by_layer


def _estimated_cost_of_split_at_layer(
    network: Sequential,
    cost_of_backsubstitution_pass_per_layer: Dict[LayerTag, float],
    previously_accumulated_cost: float,
) -> Tuple[Dict[LayerTag, float], float]:
    cost_of_split_at_layer: Dict[LayerTag, float] = {}
    accumulated_cost = previously_accumulated_cost
    for layer in reversed(network.layers):
        if (
            isinstance(layer, ReLU)
            or isinstance(layer, Sigmoid)
            or isinstance(layer, Tanh)
        ):
            accumulated_cost += cost_of_backsubstitution_pass_per_layer[
                layer_tag(layer)
            ]
            cost_of_split_at_layer[layer_tag(layer)] = accumulated_cost
        elif isinstance(layer, Sequential):
            (
                nested_cost_of_split_at_layer,
                accumulated_cost,
            ) = _estimated_cost_of_split_at_layer(
                layer, cost_of_backsubstitution_pass_per_layer, accumulated_cost
            )
            cost_of_split_at_layer = {
                **cost_of_split_at_layer,
                **nested_cost_of_split_at_layer,
            }
        elif isinstance(layer, ResidualBlock):
            (
                cost_of_split_at_layer_a,
                accumulated_cost_a,
            ) = _estimated_cost_of_split_at_layer(
                layer.path_a, cost_of_backsubstitution_pass_per_layer, accumulated_cost
            )
            (
                cost_of_split_at_layer_b,
                accumulated_cost_b,
            ) = _estimated_cost_of_split_at_layer(
                layer.path_b, cost_of_backsubstitution_pass_per_layer, accumulated_cost
            )
            cost_of_split_at_layer = {
                **cost_of_split_at_layer,
                **cost_of_split_at_layer_a,
                **cost_of_split_at_layer_b,
            }
            accumulated_cost = (
                accumulated_cost_a + accumulated_cost_b - accumulated_cost
            )
    return cost_of_split_at_layer, accumulated_cost


def _estimated_cost_of_backsubstitution_pass_per_layer(
    network: Sequential,
    cost_of_backsubstitution_operation_per_layer: Dict[LayerTag, float],
    previously_accumulated_cost: float,
) -> Tuple[Dict[LayerTag, float], float]:
    cost_of_backsubstitution_pass_per_layer: Dict[LayerTag, float] = {}
    accumulated_cost = previously_accumulated_cost
    for layer in network.layers:
        if (
            isinstance(layer, ReLU)
            or isinstance(layer, Sigmoid)
            or isinstance(layer, Tanh)
        ):
            number_of_queries = np.prod(layer.output_dim)
            cost_of_backsubstitution_pass_per_layer[layer_tag(layer)] = (
                number_of_queries * accumulated_cost
            )
            accumulated_cost += cost_of_backsubstitution_operation_per_layer[
                layer_tag(layer)
            ]
        elif isinstance(layer, Sequential):
            (
                nested_cost_of_backsubstitution_pass_per_layer,
                accumulated_cost,
            ) = _estimated_cost_of_backsubstitution_pass_per_layer(
                layer, cost_of_backsubstitution_operation_per_layer, accumulated_cost
            )
            cost_of_backsubstitution_pass_per_layer = {
                **cost_of_backsubstitution_pass_per_layer,
                **nested_cost_of_backsubstitution_pass_per_layer,
            }
        elif isinstance(layer, ResidualBlock):
            (
                cost_of_backsubstitution_pass_per_layer_a,
                accumulated_cost_a,
            ) = _estimated_cost_of_backsubstitution_pass_per_layer(
                layer.path_a,
                cost_of_backsubstitution_operation_per_layer,
                accumulated_cost,
            )
            (
                cost_of_backsubstitution_pass_per_layer_b,
                accumulated_cost_b,
            ) = _estimated_cost_of_backsubstitution_pass_per_layer(
                layer.path_b,
                cost_of_backsubstitution_operation_per_layer,
                accumulated_cost,
            )
            cost_of_backsubstitution_pass_per_layer = {
                **cost_of_backsubstitution_pass_per_layer,
                **cost_of_backsubstitution_pass_per_layer_a,
                **cost_of_backsubstitution_pass_per_layer_b,
            }
            accumulated_cost = (
                accumulated_cost_a + accumulated_cost_b - accumulated_cost
            )
        else:
            accumulated_cost += cost_of_backsubstitution_operation_per_layer[
                layer_tag(layer)
            ]

    return cost_of_backsubstitution_pass_per_layer, accumulated_cost


def _estimated_cost_of_backsubstitution_operation_per_layer(
    network: Sequential,
    prima_coefficients: Dict[LayerTag, Tuple[Tensor, Tensor, Tensor]],
) -> Dict[LayerTag, float]:
    cost_of_backsubstitution_operation_per_layer: Dict[LayerTag, float] = {}
    for layer in network.layers:
        if isinstance(layer, Sequential):
            nested_cost_of_backsubstitution_operation_per_layer = (
                _estimated_cost_of_backsubstitution_operation_per_layer(
                    layer, prima_coefficients
                )
            )
            cost_of_backsubstitution_operation_per_layer = {
                **cost_of_backsubstitution_operation_per_layer,
                **nested_cost_of_backsubstitution_operation_per_layer,
            }
        if isinstance(layer, ResidualBlock):
            cost_of_backsubstitution_operation_per_layer_a = (
                _estimated_cost_of_backsubstitution_operation_per_layer(
                    layer.path_a, prima_coefficients
                )
            )
            cost_of_backsubstitution_operation_per_layer_b = (
                _estimated_cost_of_backsubstitution_operation_per_layer(
                    layer.path_b, prima_coefficients
                )
            )
            cost_of_backsubstitution_operation_per_layer = {
                **cost_of_backsubstitution_operation_per_layer,
                **cost_of_backsubstitution_operation_per_layer_a,
                **cost_of_backsubstitution_operation_per_layer_b,
            }
        else:
            cost_of_backsubstitution_operation_per_layer[
                layer_tag(layer)
            ] = _estimated_cost_of_backsubstitution_operation(layer, prima_coefficients)

    return cost_of_backsubstitution_operation_per_layer


def _estimated_cost_of_backsubstitution_operation(
    layer: AbstractModule,
    prima_coefficients: Dict[LayerTag, Tuple[Tensor, Tensor, Tensor]],
) -> float:
    if isinstance(layer, ReLU):
        n_prima_constraints = 0
        if layer_tag(layer) in prima_coefficients:
            n_prima_constraints += prima_coefficients[layer_tag(layer)][0].shape[2]
        return np.prod(layer.output_dim) + n_prima_constraints
    elif isinstance(layer, Conv2d):
        kernel_size = layer.kernel_size[0]
        number_of_neurons = np.prod(layer.output_dim)
        return number_of_neurons * kernel_size * kernel_size
    elif isinstance(layer, Linear):
        return np.prod(layer.weight.shape)
    elif isinstance(layer, BatchNorm2d):
        return np.prod(layer.input_dim)
    elif isinstance(layer, ResidualBlock):
        return (
            _estimated_cost_of_backsubstitution_operation(
                layer.path_a, prima_coefficients
            )
            + _estimated_cost_of_backsubstitution_operation(
                layer.path_b, prima_coefficients
            )
            + np.prod(layer.output_dim)
        )
    elif isinstance(layer, Sequential):
        cost = 0.0
        for sub_layers in layer.layers:
            cost += _estimated_cost_of_backsubstitution_operation(
                sub_layers, prima_coefficients
            )
        return cost
    else:
        return 0.0
