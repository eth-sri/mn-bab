from __future__ import annotations

import random
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from src.abstract_layers.abstract_basic_block import BasicBlock
from src.abstract_layers.abstract_bn2d import BatchNorm2d
from src.abstract_layers.abstract_conv2d import Conv2d
from src.abstract_layers.abstract_linear import Linear
from src.abstract_layers.abstract_module import AbstractModule
from src.abstract_layers.abstract_network import AbstractNetwork
from src.abstract_layers.abstract_relu import ReLU
from src.abstract_layers.abstract_sequential import Sequential
from src.mn_bab_optimizer import MNBabOptimizer
from src.mn_bab_shape import MN_BaB_Shape
from src.utilities.batching import batch_layer_properties
from src.utilities.general import get_neg_pos_comp
from src.verification_subproblem import VerificationSubproblem


# adapted from: https://github.com/KaidiXu/Beta-CROWN/blob/master/src/babsr_score_conv.py
# commit hash: 77a055a39bd338367b9c335316004863681fb671
def find_index_to_split_with_babsr(
    subproblem: VerificationSubproblem,
    network: AbstractNetwork,
    query_coef: Tensor,
    split_cost_by_layer: Optional[Dict[int, float]],
    use_prima_contributions: bool,
    use_optimized_slopes: bool,
    use_beta_contributions: bool,
    propagation_effect_mode: str,
    use_indirect_effect: bool,
    lower_bound_reduce_op_identifier: str,
    use_abs: bool,
    use_active_constraint_branching_scores: bool,
) -> Tuple[int, ...]:
    assert (
        not subproblem.is_fully_split
    ), "Can't find a node to split for fully split subproblems."
    lower_bound_reduce_op = _get_lower_bound_reduce_op(lower_bound_reduce_op_identifier)

    # arguments in beta-crown implementation that always set to these values
    decision_threshold = 0.001
    activation_layer_ids = network.get_activation_layer_ids()
    sparsest_layer_id = activation_layer_ids[0]

    device = next(network.parameters()).device
    batch_size = 1

    parameters_by_starting_layer = _move_to(
        subproblem.parameters_by_starting_layer, device
    )
    split_constraints = _move_to(_clone(subproblem.split_constraints), device)
    if use_prima_contributions:
        prima_coefficients = _move_to(subproblem.prima_coefficients, device)
    else:
        prima_coefficients = None

    if use_active_constraint_branching_scores:
        score, intercept_tb = _compute_active_constraint_scores(
            network,
            split_constraints,
            parameters_by_starting_layer,
            prima_coefficients,
            batch_size,
            device,
        )
    else:
        intermediate_bounds = _move_to(subproblem.intermediate_bounds, device)

        score, intercept_tb = _compute_split_scores(
            query_coef,
            network,
            split_constraints,
            intermediate_bounds,
            parameters_by_starting_layer,
            prima_coefficients,
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

    decision: List[Tuple[int, ...]] = []
    for batch_index in range(batch_size):
        new_score = {k: score[k][batch_index] for k in score.keys()}
        max_info = {k: torch.max(new_score[k]) for k in new_score.keys()}
        decision_layer_id = sorted(
            new_score.keys(), key=lambda x: torch.max(new_score[x]), reverse=True
        )[0]
        decision_index_flattened = torch.argmax(new_score[decision_layer_id])
        decision_index = np.unravel_index(
            decision_index_flattened.cpu(), new_score[decision_layer_id].shape
        )

        if (
            decision_layer_id != sparsest_layer_id
            and max_info[decision_layer_id].item() > decision_threshold
        ):
            decision.append((decision_layer_id, *decision_index))
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

                decision.append((intercept_layer_id, *intercept_index))
            else:
                decision.append(
                    _find_random_node_to_split(batch_index, split_constraints)
                )

    assert len(decision) == batch_size
    return decision[0]


def _get_lower_bound_reduce_op(
    lower_bound_reduce_op_identifier: str,
) -> Callable[[Tensor, Tensor], Tensor]:
    if lower_bound_reduce_op_identifier == "min":
        lower_bound_reduce_op = torch.minimum
    elif lower_bound_reduce_op_identifier == "max":
        lower_bound_reduce_op = torch.maximum
    else:
        raise RuntimeError("Unknown reduce operation for branching")
    return lower_bound_reduce_op


def _compute_active_constraint_scores(
    network: AbstractNetwork,
    split_constraints: Dict[int, Tensor],
    parameters_by_starting_layer: Dict[int, Dict[str, Dict[int, Tensor]]],
    prima_coefficients: Dict[int, Tuple[Tensor, Tensor, Tensor]],
    batch_size: int,
    device: torch.device,
) -> Tuple[Dict[int, Tensor], Dict[int, Tensor]]:
    score: Dict[int, Tensor] = {}
    backup_score: Dict[int, Tensor] = {}

    return _compute_active_constraint_scores_sequential(
        score,
        backup_score,
        network,
        split_constraints,
        parameters_by_starting_layer[id(network)],
        prima_coefficients,
        batch_size,
        device,
    )


def _compute_active_constraint_scores_sequential(
    score: Dict[int, Tensor],
    backup_score: Dict[int, Tensor],
    network: Sequential,
    split_constraints: Dict[int, Tensor],
    optimizable_parameters: Dict[str, Dict[int, Tensor]],
    prima_coefficients: Dict[int, Tuple[Tensor, Tensor, Tensor]],
    batch_size: int,
    device: torch.device,
) -> Tuple[Dict[int, Tensor], Dict[int, Tensor]]:
    for layer in reversed(network.layers):
        if isinstance(layer, ReLU):
            direct_effect = _compute_active_constraint_score(
                split_constraints,
                optimizable_parameters,
                prima_coefficients,
                batch_size,
                device,
                layer,
            )
            score[id(layer)] = direct_effect.squeeze(1)
            backup_score[id(layer)] = -1 * direct_effect.squeeze(1)
        elif isinstance(layer, Sequential):
            score, backup_score = _compute_active_constraint_scores_sequential(
                score,
                backup_score,
                layer,
                split_constraints,
                optimizable_parameters,
                prima_coefficients,
                batch_size,
                device,
            )
        elif isinstance(layer, BasicBlock):
            score, backup_score = _compute_active_constraint_scores_sequential(
                score,
                backup_score,
                layer.path_a,
                split_constraints,
                optimizable_parameters,
                prima_coefficients,
                batch_size,
                device,
            )
            score, backup_score = _compute_active_constraint_scores_sequential(
                score,
                backup_score,
                layer.path_b,
                split_constraints,
                optimizable_parameters,
                prima_coefficients,
                batch_size,
                device,
            )
    return score, backup_score


@torch.no_grad()
def _compute_split_scores(
    query_coef: Tensor,
    network: AbstractNetwork,
    split_constraints: Dict[int, Tensor],
    intermediate_bounds: OrderedDict[int, Tuple[Tensor, Tensor]],
    parameters_by_starting_layer: Dict[int, Dict[str, Dict[int, Tensor]]],
    prima_coefficients: Optional[Dict[int, Tuple[Tensor, Tensor, Tensor]]],
    batch_size: int,
    device: torch.device,
    use_optimized_slopes: bool,
    use_beta_contributions: bool,
    propagation_effect_mode: str,
    use_indirect_effect: bool,
    lower_bound_reduce_op: Callable[[Tensor, Tensor], Tensor],
    use_abs: bool,
) -> Tuple[Dict[int, Tensor], Dict[int, Tensor]]:
    batch_repeats = batch_size, *([1] * (len(query_coef.shape) - 1))
    batch_query_coef = query_coef.repeat(batch_repeats).to(device)

    abstract_shape = MN_BaB_Shape(
        lb_coef=batch_query_coef,
        ub_coef=batch_query_coef,
        carried_over_optimizable_parameters=parameters_by_starting_layer,
        prima_coefficients=prima_coefficients,
        split_constraints=split_constraints,
    )
    score: Dict[int, Tensor] = {}
    contribution_fractions: Dict[int, Dict[int, Tensor]] = {}
    backup_score: Dict[int, Tensor] = {}

    network.reset_input_bounds()
    network.set_intermediate_input_bounds(intermediate_bounds)
    abstract_shape.set_optimizable_parameters(id(network))
    if not use_optimized_slopes or not use_beta_contributions:
        abstract_shape.optimizable_parameters = _clone(
            abstract_shape.optimizable_parameters
        )
    assert abstract_shape.optimizable_parameters is not None
    if not use_optimized_slopes:
        abstract_shape.optimizable_parameters = _change_alphas_to_WK_slopes_in(
            abstract_shape.optimizable_parameters, intermediate_bounds
        )
    if not use_beta_contributions:
        abstract_shape.optimizable_parameters = _set_beta_parameters_to_zero(
            abstract_shape.optimizable_parameters
        )

    score, backup_score, __ = _compute_split_scores_sequential(
        abstract_shape,
        network,
        intermediate_bounds,
        score,
        backup_score,
        contribution_fractions,
        propagation_effect_mode,
        use_indirect_effect,
        lower_bound_reduce_op,
        use_abs,
    )
    return score, backup_score


def _compute_split_scores_sequential(
    abstract_shape: MN_BaB_Shape,
    network: Sequential,
    intermediate_bounds: OrderedDict[int, Tuple[Tensor, Tensor]],
    score: Dict[int, Tensor],
    backup_score: Dict[int, Tensor],
    contribution_fractions: Dict[int, Dict[int, Tensor]],
    propagation_effect_mode: str,
    use_indirect_effect: bool,
    lower_bound_reduce_op: Callable[[Tensor, Tensor], Tensor],
    use_abs: bool,
) -> Tuple[Dict[int, Tensor], Dict[int, Tensor], Dict[int, Dict[int, Tensor]]]:
    assert abstract_shape.split_constraints is not None
    assert abstract_shape.carried_over_optimizable_parameters is not None
    for layer_idx, layer in reversed(list(enumerate(network.layers))):
        if isinstance(layer, ReLU):
            previous_layer = network.layers[layer_idx - 1]
            current_layer_lower_bounds = intermediate_bounds[id(layer)][0]
            current_layer_upper_bounds = intermediate_bounds[id(layer)][1]
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
                        network,
                        id(layer),
                        layer_idx,
                        abstract_shape.split_constraints,
                        intermediate_bounds,
                        abstract_shape.carried_over_optimizable_parameters,
                        abstract_shape.prima_coefficients,
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
                    contribution_fractions[contributing_layer_id][id(layer)] = fractions

                indirect_effect = _compute_indirect_effect(
                    contribution_fractions,
                    score,
                    id(layer),
                    current_layer_score.shape,
                    current_layer_score.device,
                )
                current_layer_score += indirect_effect
            if use_abs:
                current_layer_score = abs(current_layer_score)
            score[id(layer)] = current_layer_score.squeeze(1)
            backup_score[id(layer)] = -1 * direct_effect.squeeze(1)
            abstract_shape = layer.backsubstitute(abstract_shape)
        elif isinstance(layer, Sequential):
            (
                score,
                backup_score,
                contribution_fractions,
            ) = _compute_split_scores_sequential(
                abstract_shape,
                layer,
                intermediate_bounds,
                score,
                backup_score,
                contribution_fractions,
                propagation_effect_mode,
                use_indirect_effect,
                lower_bound_reduce_op,
                use_abs,
            )
        elif isinstance(layer, BasicBlock):
            in_lb_bias = abstract_shape.lb_bias.clone()
            in_ub_bias = abstract_shape.ub_bias.clone()
            in_lb_coef = abstract_shape.lb_coef.clone()
            in_ub_coef = abstract_shape.ub_coef.clone()

            (
                score,
                backup_score,
                contribution_fractions,
            ) = _compute_split_scores_sequential(
                abstract_shape,
                layer.path_a,
                intermediate_bounds,
                score,
                backup_score,
                contribution_fractions,
                propagation_effect_mode,
                use_indirect_effect,
                lower_bound_reduce_op,
                use_abs,
            )

            a_lb_bias = abstract_shape.lb_bias.clone()
            a_ub_bias = abstract_shape.ub_bias.clone()
            a_lb_coef = abstract_shape.lb_coef.clone()
            a_ub_coef = abstract_shape.ub_coef.clone()

            abstract_shape.update_bounds(in_lb_coef, in_ub_coef, in_lb_bias, in_ub_bias)
            (
                score,
                backup_score,
                contribution_fractions,
            ) = _compute_split_scores_sequential(
                abstract_shape,
                layer.path_b,
                intermediate_bounds,
                score,
                backup_score,
                contribution_fractions,
                propagation_effect_mode,
                use_indirect_effect,
                lower_bound_reduce_op,
                use_abs,
            )

            new_lb_bias = (
                a_lb_bias + abstract_shape.lb_bias - in_lb_bias
            )  # Both the shape in a and in b  contain the initial bias terms, so one has to be subtracted
            new_ub_bias = a_ub_bias + abstract_shape.ub_bias - in_ub_bias
            new_lb_coef = a_lb_coef + abstract_shape.lb_coef
            new_ub_coef = a_ub_coef + abstract_shape.ub_coef

            abstract_shape.update_bounds(
                new_lb_coef, new_ub_coef, new_lb_bias, new_ub_bias
            )  # TODO look at merging of dependence sets
        else:
            abstract_shape = layer.backsubstitute(abstract_shape)

    return score, backup_score, contribution_fractions


def _change_alphas_to_WK_slopes_in(
    optimizable_parameters: Dict[str, Dict[int, Tensor]],
    intermediate_bounds: OrderedDict[int, Tuple[Tensor, Tensor]],
) -> Dict[str, Dict[int, Tensor]]:
    for param_key, layer_parameters in optimizable_parameters.items():
        if "alpha" in param_key:
            for layer_id, parameters in optimizable_parameters[param_key].items():
                current_layer_lower_bounds = intermediate_bounds[layer_id][0]
                current_layer_upper_bounds = intermediate_bounds[layer_id][1]
                ub_slope, __ = _babsr_ratio_computation(
                    current_layer_lower_bounds, current_layer_upper_bounds
                )
                WK_slopes = ub_slope
                optimizable_parameters[param_key][layer_id] = WK_slopes
    return optimizable_parameters


def _set_beta_parameters_to_zero(
    optimizable_parameters: Dict[str, Dict[int, Tensor]],
) -> Dict[str, Dict[int, Tensor]]:
    for param_key, layer_parameters in optimizable_parameters.items():
        if "beta" in param_key:
            for layer_id, parameters in optimizable_parameters[param_key].items():
                optimizable_parameters[param_key][layer_id] = torch.zeros_like(
                    parameters
                )
    return optimizable_parameters


def _babsr_ratio_computation(
    lower_bound: Tensor, upper_bound: Tensor
) -> Tuple[Tensor, Tensor]:
    lower_temp = lower_bound - F.relu(lower_bound)
    upper_temp = F.relu(upper_bound)
    slope_ratio = upper_temp / (upper_temp - lower_temp)
    intercept = -1 * lower_temp * slope_ratio

    return slope_ratio.nan_to_num(), intercept.nan_to_num()


def _compute_active_constraint_score(
    split_constraints: Dict[int, Tensor],
    optimizable_parameters: Dict[str, Dict[int, Tensor]],
    prima_coefficients: Dict[int, Tuple[Tensor, Tensor, Tensor]],
    batch_size: int,
    device: torch.device,
    layer: ReLU,
) -> Tensor:
    unstable_nodes_mask = (split_constraints[id(layer)] == 0).unsqueeze(1)
    if (
        prima_coefficients is None
        or id(layer) not in prima_coefficients
        or prima_coefficients[id(layer)][0].shape[2] == 0
    ):
        return torch.zeros(batch_size, 1, *layer.output_dim, device=device)

    (
        current_layer_prima_output_coefficients,
        current_layer_prima_input_coefficients,
        __,
    ) = prima_coefficients[id(layer)]
    prima_parameters = optimizable_parameters["prima_lb"][id(layer)]
    prima_output_contribution = layer._mulitply_prima_coefs_and_parameters(
        torch.sqrt(
            torch.square(current_layer_prima_output_coefficients)
        ),  # abs not available for sparse tensors
        prima_parameters,
    )
    prima_input_contribution = layer._mulitply_prima_coefs_and_parameters(
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
    propagation_effect_mode: str,
    lower_bound_reduce_op: Callable[[Tensor, Tensor], Tensor],
) -> Tuple[Tensor, Tensor]:
    assert abstract_shape.optimizable_parameters is not None
    assert abstract_shape.split_constraints is not None
    if id(layer) in abstract_shape.prima_coefficients:
        (
            current_layer_prima_output_coefficients,
            current_layer_prima_input_coefficients,
            __,
        ) = abstract_shape.prima_coefficients[id(layer)]
        prima_parameters = abstract_shape.optimizable_parameters["prima_lb"][id(layer)]
        prima_output_contribution = layer._mulitply_prima_coefs_and_parameters(
            current_layer_prima_output_coefficients, prima_parameters
        )
        prima_input_contribution = layer._mulitply_prima_coefs_and_parameters(
            current_layer_prima_input_coefficients, prima_parameters
        )
    else:
        prima_output_contribution = torch.zeros_like(abstract_shape.lb_coef)
        prima_input_contribution = torch.zeros_like(abstract_shape.lb_coef)

    lb_coef_before_relaxation = abstract_shape.lb_coef + prima_output_contribution

    lb_slope = abstract_shape.optimizable_parameters["alpha_lb"][id(layer)]
    ub_slope, ub_intercept = _babsr_ratio_computation(
        current_layer_lower_bounds, current_layer_upper_bounds
    )
    lb_slope, ub_slope, ub_intercept = (
        lb_slope.unsqueeze(1),
        ub_slope.unsqueeze(1),
        ub_intercept.unsqueeze(1),
    )

    (
        neg_lb_coef_before_relaxation,
        pos_lb_coef_before_relaxation,
    ) = get_neg_pos_comp(lb_coef_before_relaxation)

    beta_parameters = abstract_shape.optimizable_parameters["beta_lb"][id(layer)]
    beta_contribution_shape = (abstract_shape.batch_size, 1, *layer.output_dim)
    beta_contribution = (
        beta_parameters * abstract_shape.split_constraints[id(layer)]
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
    unstable_nodes_mask = (abstract_shape.split_constraints[id(layer)] == 0).unsqueeze(
        1
    )
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
        previous_layer_bias = previous_layer.bias
    # unsqueeze to batch_dim, query_dim, bias_dim
    previous_layer_bias = previous_layer_bias.unsqueeze(0).unsqueeze(0)
    expected_number_of_coef_dims_if_prev_layer_is_conv = 5
    if coef_dim == expected_number_of_coef_dims_if_prev_layer_is_conv:
        # unsqueeze bias from batch_dim, query_sim, channel to batch_dim, query_sim, channel, height, width
        previous_layer_bias = previous_layer_bias.unsqueeze(-1).unsqueeze(-1)
    assert coef_dim == previous_layer_bias.dim(), "bias expanded to unexpected shape"
    return previous_layer_bias


def _compute_direct_and_propagation_effect_on_upper_bound(
    abstract_shape: MN_BaB_Shape,
    layer: ReLU,
    previous_layer: Union[Linear, Conv2d],
    current_layer_lower_bounds: Tensor,
    current_layer_upper_bounds: Tensor,
    propagation_effect_mode: str,
    lower_bound_reduce_op: Callable[[Tensor, Tensor], Tensor],
) -> Tuple[Tensor, Tensor]:
    assert abstract_shape.optimizable_parameters is not None
    assert abstract_shape.split_constraints is not None
    if id(layer) in abstract_shape.prima_coefficients:
        (
            current_layer_prima_output_coefficients,
            current_layer_prima_input_coefficients,
            __,
        ) = abstract_shape.prima_coefficients[id(layer)]
        prima_parameters = abstract_shape.optimizable_parameters["prima_ub"][id(layer)]
        prima_output_contribution = layer._mulitply_prima_coefs_and_parameters(
            current_layer_prima_output_coefficients, prima_parameters
        )
        prima_input_contribution = layer._mulitply_prima_coefs_and_parameters(
            current_layer_prima_input_coefficients, prima_parameters
        )
    else:
        prima_output_contribution = torch.zeros_like(abstract_shape.ub_coef)
        prima_input_contribution = torch.zeros_like(abstract_shape.ub_coef)

    ub_coef_before_relaxation = abstract_shape.ub_coef - prima_output_contribution

    lb_slope = abstract_shape.optimizable_parameters["alpha_ub"][id(layer)]
    ub_slope, ub_intercept = _babsr_ratio_computation(
        current_layer_lower_bounds, current_layer_upper_bounds
    )
    lb_slope, ub_slope, ub_intercept = (
        lb_slope.unsqueeze(1),
        ub_slope.unsqueeze(1),
        ub_intercept.unsqueeze(1),
    )

    (
        neg_ub_coef_before_relaxation,
        pos_ub_coef_before_relaxation,
    ) = get_neg_pos_comp(ub_coef_before_relaxation)

    beta_parameters = abstract_shape.optimizable_parameters["beta_ub"][id(layer)]
    beta_contribution_shape = (abstract_shape.batch_size, 1, *layer.output_dim)
    beta_contribution = (
        beta_parameters * abstract_shape.split_constraints[id(layer)]
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
    unstable_nodes_mask = (abstract_shape.split_constraints[id(layer)] == 0).unsqueeze(
        1
    )
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
    propagation_effect_mode: str,
    for_lower_bound: bool = True,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    zero = torch.zeros_like(layer_lower_bounds)
    if propagation_effect_mode == "none":
        return zero, zero, zero, zero, zero, zero
    elif propagation_effect_mode == "bias":
        return (
            prev_layer_bias,
            prev_layer_bias,
            prev_layer_bias,
            prev_layer_bias,
            prev_layer_bias,
            prev_layer_bias,
        )
    elif propagation_effect_mode == "intermediate_concretization":
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
    network: Sequential,
    starting_layer_id: int,
    starting_layer_index: int,
    split_constraints: Dict[int, Tensor],
    intermediate_bounds: OrderedDict[int, Tuple[Tensor, Tensor]],
    parameters_by_starting_layer: Dict[int, Dict[str, Dict[int, Tensor]]],
    prima_coefficients: Optional[Dict[int, Tuple[Tensor, Tensor, Tensor]]],
    batch_size: int,
    propagation_effect_mode: str,
    lower_bound_reduce_op: Callable[[Tensor, Tensor], Tensor],
) -> Dict[int, Tensor]:
    assert id(network.layers[starting_layer_index]) == starting_layer_id
    assert isinstance(network.layers[starting_layer_index], ReLU)
    layer_shape = network.layers[starting_layer_index - 1].output_dim
    device = intermediate_bounds[starting_layer_id][0].device
    abstract_shape = MN_BaB_Shape.construct_to_bound_all_outputs(
        device,
        layer_shape,
        batch_size,
        parameters_by_starting_layer,
        prima_coefficients,
        None,
        split_constraints,
        None,
    )

    contribution_fraction_to_starting_layer = {}

    starting_layer_lower_bounds = intermediate_bounds[starting_layer_id][0]
    starting_layer_upper_bounds = intermediate_bounds[starting_layer_id][1]
    abstract_shape.set_optimizable_parameters(starting_layer_id)
    assert abstract_shape.optimizable_parameters is not None
    assert abstract_shape.split_constraints is not None
    for layer_idx, layer in reversed(
        list(enumerate(network.layers[:starting_layer_index]))
    ):
        if isinstance(layer, ReLU):
            current_layer_lower_bounds = intermediate_bounds[id(layer)][0]
            current_layer_upper_bounds = intermediate_bounds[id(layer)][1]

            assert isinstance(abstract_shape.ub_coef, Tensor)
            starting_and_affected_node_unstable_mask = (
                abstract_shape.split_constraints[id(layer)] == 0
            ).unsqueeze(1) * _reshape_layer_values(
                abstract_shape.split_constraints[starting_layer_id] == 0,
                len(abstract_shape.ub_coef.shape),
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
                id(layer)
            ] = _compute_triangle_relaxation_area_change(
                starting_layer_lower_bounds,
                starting_layer_upper_bounds,
                lb_contribution,
                ub_contribution,
            )

        abstract_shape = layer.backsubstitute(abstract_shape)
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
    contribution_fractions: Dict[int, Dict[int, Tensor]],
    score: Dict[int, Tensor],
    current_layer_id: int,
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
    batch_index: int, split_constraints: Dict[int, Tensor]
) -> Tuple[int, ...]:
    for layer_id in reversed(list(split_constraints.keys())):
        try:
            nodes_not_yet_split_in_layer = split_constraints[layer_id] == 0
            unstable_neuron_indices_in_layer = torch.nonzero(
                nodes_not_yet_split_in_layer[batch_index]
            )
            n_unstable_neurons = unstable_neuron_indices_in_layer.shape[0]
            random_unstable_neuron_index = random.randint(0, n_unstable_neurons - 1)
            unstable_relu_index = tuple(
                unstable_neuron_indices_in_layer[random_unstable_neuron_index].tolist(),
            )

            break
        except ValueError:
            continue

    return (layer_id, *unstable_relu_index)


def _adjust_based_on_cost(
    scores: Dict[int, Tensor],
    backup_scores: Dict[int, Tensor],
    split_cost_by_layer: Dict[int, float],
) -> Tuple[Dict[int, Tensor], Dict[int, Tensor]]:
    for layer_id, split_cost in split_cost_by_layer.items():
        assert layer_id in scores
        assert layer_id in backup_scores
        scores[layer_id] = scores[layer_id] / split_cost
        backup_scores[layer_id] = backup_scores[layer_id] / split_cost
    return scores, backup_scores


# adapted from: https://github.com/huanzhang12/alpha-beta-CROWN/blob/main/src/branching_heuristics.py
# commit hash: cdbcba0ea346ebd03d552023773829fe6e0822c7
def find_index_to_split_with_filtered_smart_branching(
    subproblem: VerificationSubproblem,
    optimizer: MNBabOptimizer,
    network: AbstractNetwork,
    query_coef: Tensor,
    split_cost_by_layer: Optional[Dict[int, float]],
    input_lb: Tensor,
    input_ub: Tensor,
    number_of_preselected_candidates_per_layer: int,
    lower_bound_reduce_op_identifier: str,
    batch_sizes: Sequence[int],
    recompute_intermediate_bounds_after_branching: bool,
) -> Tuple[int, ...]:
    assert (
        not subproblem.is_fully_split
    ), "Can't find a node to split for fully split subproblems."
    lower_bound_reduce_op = _get_lower_bound_reduce_op(lower_bound_reduce_op_identifier)
    batch_sizes_by_layer_id = {
        layer_id: batch_sizes[layer_index]
        for layer_index, layer_id in enumerate(subproblem.intermediate_bounds.keys())
    }

    device = next(network.parameters()).device
    batch_size = 1
    nodes_not_yet_split_mask = {
        layer_id: (
            (subproblem.intermediate_bounds[layer_id][0] < 0)
            & (subproblem.intermediate_bounds[layer_id][1] > 0)
            & (layer_split_constraints == 0)
        )
        .float()
        .to(device)
        for layer_id, layer_split_constraints in subproblem.split_constraints.items()
    }

    split_constraints = _move_to(_clone(subproblem.split_constraints), device)
    intermediate_bounds = _move_to(subproblem.intermediate_bounds, device)
    parameters_by_starting_layer = _move_to(
        subproblem.parameters_by_starting_layer, device
    )
    babsr_scores, intercept_tb = _compute_split_scores(
        query_coef,
        network,
        split_constraints,
        intermediate_bounds,
        parameters_by_starting_layer,
        None,
        batch_size,
        device,
        use_optimized_slopes=False,
        use_beta_contributions=False,
        propagation_effect_mode="bias",
        use_indirect_effect=False,
        lower_bound_reduce_op=lower_bound_reduce_op,
        use_abs=True,
    )

    decision: List[Tuple[int, ...]] = []
    for batch_index in range(batch_size):
        babsr_scores_of_batch_element = {
            k: babsr_scores[k][batch_index] for k in babsr_scores.keys()
        }
        intercept_tb_of_batch_element = {
            k: intercept_tb[k][batch_index] for k in intercept_tb.keys()
        }

        all_candidates = {}
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
                (layer_id, *candidate_index)
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


def _clone(obj: Any) -> Any:
    if torch.is_tensor(obj):
        return obj.clone()
    elif isinstance(obj, dict):
        res: Any = {}
        for k, v in obj.items():
            res[k] = _clone(v)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(_clone(v))
        return res
    elif isinstance(obj, tuple):
        res = []
        for v in obj:
            res.append(_clone(v))
        return tuple(res)
    else:
        raise TypeError("Invalid type for clone")


def _move_to(obj: Any, device: torch.device) -> Any:
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        res: Any = {}
        for k, v in obj.items():
            res[k] = _move_to(v, device)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(_move_to(v, device))
        return res
    elif isinstance(obj, tuple):
        res = []
        for v in obj:
            res.append(_move_to(v, device))
        return tuple(res)
    else:
        raise TypeError("Invalid type for move_to")


def _get_indices_of_topk(x: Tensor, k: int, largest: bool) -> List[Tuple[int, ...]]:
    flattenend_indices = torch.topk(x.flatten(), k, largest=largest).indices.cpu()
    indices_by_dimension = np.unravel_index(flattenend_indices, x.shape)
    return [tuple(indices[i] for indices in indices_by_dimension) for i in range(k)]


def _compute_candidate_scores_for(
    candidate_nodes_to_split: Sequence[Tuple[int, ...]],
    subproblem: VerificationSubproblem,
    optimizer: MNBabOptimizer,
    query_coef: Tensor,
    network: AbstractNetwork,
    input_lb: Tensor,
    input_ub: Tensor,
    lower_bound_reduce_op: Callable[[Tensor, Tensor], Tensor],
    batch_size_for_bounding: int,
    recompute_intermediate_bounds_after_branching: bool,
) -> Sequence[float]:
    max_queries = 2 * batch_size_for_bounding
    subproblems_to_bound = [
        split
        for node_to_split in candidate_nodes_to_split
        for split in subproblem.split(
            node_to_split, recompute_intermediate_bounds_after_branching
        )
    ]
    n_scores_to_compute = len(subproblems_to_bound)
    device = input_lb.device
    for subproblem in subproblems_to_bound:
        subproblem.to(device)

    candidate_scores: List[float] = []
    offset = 0
    while offset < n_scores_to_compute:
        subproblem_batch = batch_layer_properties(
            subproblems_to_bound[offset : offset + max_queries]
        )

        batch_repeats = min(offset + max_queries, n_scores_to_compute) - offset, *(
            [1] * (len(query_coef.shape) - 1)
        )
        lower_bounds, __ = optimizer.bound_minimum_with_deep_poly(
            query_coef.to(device).repeat(batch_repeats),
            network,
            input_lb,
            input_ub,
            subproblem_batch.split_constraints,
            subproblem_batch.intermediate_layer_bounds_to_be_kept_fixed,
            subproblem_batch.intermediate_bounds,
            subproblem_batch.parameters_by_starting_layer,
            subproblem_batch.prima_coefficients,
        )
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
    scores: Dict[Tuple[int, ...], float], split_cost_by_layer: Dict[int, float]
) -> Dict[Tuple[int, ...], float]:
    for node_to_split, score in scores.items():
        layer_id = node_to_split[0]
        assert layer_id in split_cost_by_layer
        scores[node_to_split] = score / split_cost_by_layer[layer_id]
    return scores


def compute_split_cost_by_layer(
    network: AbstractNetwork,
    prima_coefficients: Dict[int, Tuple[Tensor, Tensor, Tensor]],
    recompute_intermediate_bounds_after_branching: bool,
) -> Dict[int, float]:
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
    cost_of_backsubstitution_pass_starting_at[id(network)] = (
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
    cost_of_backsubstitution_pass_per_layer: Dict[int, float],
    previously_accumulated_cost: float,
) -> Tuple[Dict[int, float], float]:
    cost_of_split_at_layer: Dict[int, float] = {}
    accumulated_cost = previously_accumulated_cost
    for layer in reversed(network.layers):
        if isinstance(layer, ReLU):
            accumulated_cost += cost_of_backsubstitution_pass_per_layer[id(layer)]
            cost_of_split_at_layer[id(layer)] = accumulated_cost
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
        elif isinstance(layer, BasicBlock):
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
    cost_of_backsubstitution_operation_per_layer: Dict[int, float],
    previously_accumulated_cost: float,
) -> Tuple[Dict[int, float], float]:
    cost_of_backsubstitution_pass_per_layer: Dict[int, float] = {}
    accumulated_cost = previously_accumulated_cost
    for layer in network.layers:
        if isinstance(layer, ReLU):
            number_of_queries = np.prod(layer.output_dim)
            cost_of_backsubstitution_pass_per_layer[id(layer)] = (
                number_of_queries * accumulated_cost
            )
            accumulated_cost += cost_of_backsubstitution_operation_per_layer[id(layer)]
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
        elif isinstance(layer, BasicBlock):
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
            accumulated_cost += cost_of_backsubstitution_operation_per_layer[id(layer)]

    return cost_of_backsubstitution_pass_per_layer, accumulated_cost


def _estimated_cost_of_backsubstitution_operation_per_layer(
    network: Sequential,
    prima_coefficients: Dict[int, Tuple[Tensor, Tensor, Tensor]],
) -> Dict[int, float]:
    cost_of_backsubstitution_operation_per_layer: Dict[int, float] = {}
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
        if isinstance(layer, BasicBlock):
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
                id(layer)
            ] = _estimated_cost_of_backsubstitution_operation(layer, prima_coefficients)

    return cost_of_backsubstitution_operation_per_layer


def _estimated_cost_of_backsubstitution_operation(
    layer: AbstractModule,
    prima_coefficients: Dict[int, Tuple[Tensor, Tensor, Tensor]],
) -> float:
    if isinstance(layer, ReLU):
        n_prima_constraints = 0
        if id(layer) in prima_coefficients:
            n_prima_constraints += prima_coefficients[id(layer)][0].shape[2]
        return np.prod(layer.output_dim) + n_prima_constraints
    elif isinstance(layer, Conv2d):
        kernel_size = layer.kernel_size[0]
        number_of_neurons = np.prod(layer.output_dim)
        return number_of_neurons * kernel_size * kernel_size
    elif isinstance(layer, Linear):
        return np.prod(layer.weight.shape)
    elif isinstance(layer, BatchNorm2d):
        return np.prod(layer.input_dim)
    elif isinstance(layer, BasicBlock):
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
