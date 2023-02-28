import time
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim
from torch import Tensor

from src.exceptions.verification_timeout import VerificationTimeoutException
from src.mn_bab_shape import MN_BaB_Shape, num_queries
from src.state.subproblem_state import ReadonlySubproblemState
from src.state.tags import QueryTag
from src.utilities.config import IntermediateBoundOptimizationConfig
from src.utilities.queries import QueryCoef


def optimize_params_for_interm_bounds(  # noqa: C901
    query_id: QueryTag,
    query_coef: QueryCoef,
    subproblem_state: ReadonlySubproblemState,
    opt_callback: Callable[
        [QueryCoef],
        Tuple[
            Optional[MN_BaB_Shape], Tuple[Tensor, Tensor]
        ],  # TODO: can we get rid of MN_BaB_Shape return value?
    ],
    config: IntermediateBoundOptimizationConfig,
    timeout: float = float("inf"),
) -> Tuple[Tensor, Tensor]:

    prima_lr = config.prima_lr
    alpha_lr = config.alpha_lr
    factor = config.lr_config.peak_scaling_factor
    div_factor = config.lr_config.final_div_factor
    optimization_iterations = config.optimization_iterations
    indiv_optim = config.indiv_optim
    adapt_optim = config.adapt_optim
    num_layers_to_optimize = config.num_layers_to_optimize
    optimize_prior_bounds = config.optimize_prior_bounds
    selected_query_id: Optional[QueryTag] = query_id

    # indiv_optim = True
    # optimize_prior_bounds = True
    # optimization_iterations = 50
    adapt_optim = True

    start_time = time.time()

    assert query_coef.is_leaf
    # First run for optimizable parameters
    print(f"Query {query_id}")
    prop_shape, dp_bounds = opt_callback(query_coef)
    print(f"First prop {query_id}")

    if prop_shape is None:
        return dp_bounds

    if optimize_prior_bounds:
        selected_query_id = None

    current_relu_position = len(  # TODO: get rid of this hack (maybe will conflict with other kinds of parameter sharing)
        subproblem_state.parameters.parameters_by_query
    )  # First relu has index 1

    (
        all_alpha_parameters,
        all_beta_parameters,
        all_prima_parameters,
        alpha_relu_parameters,
    ) = prop_shape.get_optimizable_parameters(selected_query_id)

    if len(all_alpha_parameters) == 0 or current_relu_position > num_layers_to_optimize:
        return dp_bounds

    parameters_to_optimize = [
        {"params": all_alpha_parameters, "lr": alpha_lr},
        {"params": all_prima_parameters, "lr": prima_lr},
    ]

    best_lower_bounds = dp_bounds[0].detach().clone()
    best_upper_bounds = dp_bounds[1].detach().clone()

    for o in range(2):
        if indiv_optim:
            for j in range(num_queries(query_coef)):
                optimizer = optim.Adam(parameters_to_optimize)
                scheduler = optim.lr_scheduler.OneCycleLR(  # type: ignore[attr-defined]
                    optimizer,
                    (alpha_lr * factor, prima_lr * factor),
                    optimization_iterations,
                    final_div_factor=div_factor,
                )

                for alpha_parameters in alpha_relu_parameters:
                    alpha_parameters.data = torch.zeros_like(
                        alpha_parameters.data, requires_grad=True
                    )

                for prima_parameters in all_prima_parameters:
                    prima_parameters.data = torch.zeros_like(
                        prima_parameters.data, requires_grad=True
                    )

                for i in range(optimization_iterations):
                    if time.time() - start_time > timeout:
                        raise VerificationTimeoutException()
                    # @Robin Set specific query coeff
                    prop_shape, (output_lbs, output_ubs) = opt_callback(query_coef)

                    best_lower_bounds = torch.maximum(
                        output_lbs, best_lower_bounds
                    ).detach()
                    best_upper_bounds = torch.minimum(
                        output_ubs, best_upper_bounds
                    ).detach()

                    if o == 0:
                        loss = -output_lbs[0][j]
                    else:
                        loss = output_ubs[0][j]

                    # print(loss)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    # Clamping only for ReLU
                    for alpha_parameters in alpha_relu_parameters:
                        alpha_parameters.data = torch.clamp(
                            alpha_parameters.data, 0.0, 1.0
                        )

                    for prima_parameters in all_prima_parameters:
                        prima_parameters.data = torch.clamp(
                            prima_parameters.data, min=0.0
                        )
        elif adapt_optim:
            assert isinstance(query_coef, Tensor)
            if len(query_coef.shape) == 5:  # We have a channel dimension
                num_channels = query_coef.shape[2]
                neurons_per_channel = np.prod(query_coef.shape[3:])
            else:
                num_channels = 1
                neurons_per_channel = np.prod(query_coef.shape[2:])

            assert num_queries(query_coef) == num_channels * neurons_per_channel

            for j in range(num_channels):

                curr_query_start = j * neurons_per_channel

                optimizer = optim.Adam(parameters_to_optimize)
                scheduler = optim.lr_scheduler.OneCycleLR(  # type: ignore[attr-defined]
                    optimizer,
                    (alpha_lr * factor, prima_lr * factor),
                    optimization_iterations,
                    final_div_factor=div_factor,
                )

                for alpha_parameters in alpha_relu_parameters:
                    alpha_parameters.data = torch.zeros_like(
                        alpha_parameters.data, requires_grad=True
                    )

                for prima_parameters in all_prima_parameters:
                    prima_parameters.data = torch.zeros_like(
                        prima_parameters.data, requires_grad=True
                    )

                for i in range(optimization_iterations):
                    if time.time() - start_time > timeout:
                        raise VerificationTimeoutException()

                    curr_query_coef = query_coef[
                        :, curr_query_start : curr_query_start + neurons_per_channel, :
                    ]

                    prop_shape, (output_lbs, output_ubs) = opt_callback(curr_query_coef)

                    best_lower_bounds[
                        0, curr_query_start : curr_query_start + neurons_per_channel
                    ] = torch.maximum(
                        output_lbs[0],
                        best_lower_bounds[
                            0, curr_query_start : curr_query_start + neurons_per_channel
                        ],
                    ).detach()
                    best_upper_bounds[
                        0, curr_query_start : curr_query_start + neurons_per_channel
                    ] = torch.minimum(
                        output_ubs[0],
                        best_upper_bounds[
                            0, curr_query_start : curr_query_start + neurons_per_channel
                        ],
                    ).detach()

                    if o == 0:
                        loss = -torch.sum(output_lbs)
                    else:
                        loss = torch.sum(output_ubs)

                    # print(loss)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    # Clamping only for ReLU
                    for alpha_parameters in alpha_relu_parameters:
                        alpha_parameters.data = torch.clamp(
                            alpha_parameters.data, 0.0, 1.0
                        )

                    for prima_parameters in all_prima_parameters:
                        prima_parameters.data = torch.clamp(
                            prima_parameters.data, min=0.0
                        )
        else:
            optimizer = optim.Adam(parameters_to_optimize)
            scheduler = optim.lr_scheduler.OneCycleLR(  # type: ignore[attr-defined]
                optimizer,
                (alpha_lr * factor, prima_lr * factor),
                optimization_iterations,
                final_div_factor=div_factor,
            )

            for alpha_parameters in alpha_relu_parameters:
                alpha_parameters.data = torch.zeros_like(
                    alpha_parameters.data, requires_grad=True
                )

            for prima_parameters in all_prima_parameters:
                prima_parameters.data = torch.zeros_like(
                    prima_parameters.data, requires_grad=True
                )

            for i in range(optimization_iterations):
                if time.time() - start_time > timeout:
                    raise VerificationTimeoutException()
                # @Robin Set specific query coeff
                prop_shape, (output_lbs, output_ubs) = opt_callback(query_coef)
                # TODO @Robin set
                output_lbs = output_lbs.flatten()
                output_ubs = output_ubs.flatten()

                best_lower_bounds = torch.maximum(
                    output_lbs, best_lower_bounds
                ).detach()
                best_upper_bounds = torch.minimum(
                    output_ubs, best_upper_bounds
                ).detach()

                if o == 0:
                    loss = -output_lbs.sum()
                else:
                    loss = output_ubs.sum()

                # print(loss)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                # Clamping only for ReLU
                for alpha_parameters in alpha_relu_parameters:
                    alpha_parameters.data = torch.clamp(alpha_parameters.data, 0.0, 1.0)

                for prima_parameters in all_prima_parameters:
                    prima_parameters.data = torch.clamp(prima_parameters.data, min=0.0)

    # for j in range(num_queries(query_coef)):
    #     print(
    #         f"{j} LB : {best_lower_bounds[0][j] - dp_bounds[0][0][j]:.4f} UB {dp_bounds[1][0][j] - best_upper_bounds[0][j]:.4f}"
    #     )

    return (
        best_lower_bounds,
        best_upper_bounds,
    )
