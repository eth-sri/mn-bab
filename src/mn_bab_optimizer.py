from __future__ import annotations

import time
from collections import OrderedDict
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, optim
from torch.optim import Optimizer

from src.abstract_layers.abstract_module import INFEASIBILITY_CHECK_TOLERANCE
from src.abstract_layers.abstract_network import AbstractNetwork
from src.exceptions.verification_timeout import VerificationTimeoutException
from src.mn_bab_shape import MN_BaB_Shape
from src.utilities.batching import (
    unbatch_layer_bounds,
    unbatch_layer_property,
    unbatch_parameters,
    unbatch_prima_coefficients,
)
from src.utilities.general import any_smaller
from src.verification_subproblem import VerificationSubproblem

DEFAULT_PRIMA_HYPERPARAMETERS = {
    "sparse_n": 50,
    "K": 3,
    "s": 1,
    "num_proc_to_compute_constraints": 2,
    "max_number_of_parallel_input_constraint_queries": 10000,
    "max_unstable_nodes_considered_per_layer": 1000,
    "min_relu_transformer_area_to_be_considered": 0.05,
    "fraction_of_constraints_to_keep": 1.0,
}


class MNBabOptimizer:
    def __init__(
        self,
        optimize_alpha: bool = False,
        alpha_lr: float = 0.1,
        alpha_opt_iterations: int = 20,
        optimize_prima: bool = False,
        prima_lr: float = 0.01,
        prima_opt_iterations: int = 20,
        prima_hyperparamters: Dict[str, float] = DEFAULT_PRIMA_HYPERPARAMETERS,
        peak_lr_scaling_factor: float = 2.0,
        final_lr_div_factor: float = 1e1,
        beta_lr: float = 0.05,
        use_dependence_sets: bool = False,
        use_early_termination: bool = False,
    ) -> None:
        assert not (
            optimize_prima and not optimize_alpha
        ), "If you optimize prima constraints, you also have to optimize alpha."

        self.optimize_alpha = optimize_alpha
        self.alpha_lr = alpha_lr
        self.alpha_opt_iterations = alpha_opt_iterations

        self.optimize_prima = optimize_prima
        self.prima_lr = prima_lr
        self.prima_opt_iterations = prima_opt_iterations
        self.prima_hyperparamters = prima_hyperparamters

        self.peak_lr_scaling_factor = peak_lr_scaling_factor
        self.final_lr_div_factor = final_lr_div_factor

        self.beta_lr = beta_lr

        self.use_dependence_sets = use_dependence_sets
        self.use_early_termination = use_early_termination

    def bound_root_subproblem(
        self,
        query_coef: Tensor,
        network: AbstractNetwork,
        input_lb: Tensor,
        input_ub: Tensor,
        early_stopping_threshold: float = float("inf"),
        timeout: float = float("inf"),
        device: torch.device = torch.device("cpu"),
    ) -> VerificationSubproblem:
        start_time = time.time()
        assert query_coef.shape[0] == 1, "Expected single query for root subproblem."

        root_subproblem = VerificationSubproblem.create_default(
            split_constraints=network.get_default_split_constraints()
        )
        root_subproblem.to(device)
        query_coef = query_coef.to(device)

        deep_poly_lbs, deep_poly_ubs = self.bound_minimum_with_deep_poly(
            query_coef, network, input_lb, input_ub
        )
        assert len(deep_poly_lbs) == len(deep_poly_lbs) == 1
        print("deep poly lower bounds:", deep_poly_lbs)

        # root node is never infeasible
        invalid_bounds_mask_root: Sequence[bool] = [False]
        if self._can_stop_early(
            deep_poly_lbs,
            deep_poly_ubs,
            early_stopping_threshold,
            invalid_bounds_mask_root,
        ):
            return VerificationSubproblem.create_default(
                deep_poly_lbs[0], deep_poly_ubs[0]
            )
        assert self.optimize_alpha
        time_remaining = timeout - (time.time() - start_time)
        (
            alpha_lbs,
            alpha_ubs,
            alpha_optimized_parameters,
            alpha_intermediate_bounds,
        ) = self._bound_minimum_optimizing_alpha(
            query_coef,
            network,
            input_lb,
            input_ub,
            root_subproblem.split_constraints,
            root_subproblem.intermediate_layer_bounds_to_be_kept_fixed,
            root_subproblem.intermediate_bounds,
            root_subproblem.parameters_by_starting_layer,
            early_stopping_threshold,
            time_remaining,
        )
        assert len(alpha_lbs) == len(alpha_ubs) == 1
        print("alpha lower bounds:", alpha_lbs)

        if self._can_stop_early(
            alpha_lbs,
            alpha_ubs,
            early_stopping_threshold,
            invalid_bounds_mask_root,
        ):
            return VerificationSubproblem.create_default(
                alpha_lbs[0],
                alpha_ubs[0],
            )

        intermediate_bounds = alpha_intermediate_bounds
        optimized_parameters = alpha_optimized_parameters
        prima_coefficients: Dict[int, Tuple[Tensor, Tensor, Tensor]] = {}
        if self.optimize_prima:
            time_remaining = timeout - (time.time() - start_time)
            (
                prima_lbs,
                prima_ubs,
                prima_optimized_parameters,
                prima_intermediate_bounds,
                prima_coefficients,
            ) = self._bound_minimum_optimizing_alpha_prima(
                query_coef,
                network,
                input_lb,
                input_ub,
                alpha_optimized_parameters,
                root_subproblem.split_constraints,
                root_subproblem.intermediate_layer_bounds_to_be_kept_fixed,
                intermediate_bounds,
                root_subproblem.prima_coefficients,
                early_stopping_threshold=early_stopping_threshold,
                timeout=time_remaining,
            )
            assert len(prima_lbs) == len(prima_ubs) == 1
            print("prima lower bounds:", prima_lbs)

            self._update_best_intermediate_bounds(
                intermediate_bounds,
                prima_intermediate_bounds,
            )

            if self._can_stop_early(
                prima_lbs,
                prima_ubs,
                early_stopping_threshold,
                invalid_bounds_mask_root,
            ):
                return VerificationSubproblem.create_default(prima_lbs[0], prima_ubs[0])

            optimized_parameters = prima_optimized_parameters

        best_lbs = deep_poly_lbs
        best_ubs = deep_poly_ubs
        if self.optimize_alpha:
            best_lbs = np.maximum(alpha_lbs, best_lbs).tolist()
            best_ubs = np.minimum(alpha_ubs, best_ubs).tolist()
        if self.optimize_prima:
            best_lbs = np.maximum(prima_lbs, best_lbs).tolist()
            best_ubs = np.minimum(prima_ubs, best_ubs).tolist()
        return VerificationSubproblem(
            best_lbs[0],
            best_ubs[0],
            root_subproblem.split_constraints,
            [],
            intermediate_bounds,
            optimized_parameters,
            prima_coefficients,
            invalid_bounds_mask_root[0],
            root_subproblem.number_of_nodes_split,
        )

    def improve_subproblem_batch_bounds(
        self,
        query_coef: Tensor,
        network: AbstractNetwork,
        input_lb: Tensor,
        input_ub: Tensor,
        subproblem_batch: VerificationSubproblem,
        early_stopping_threshold: float = float("inf"),
        timeout: float = float("inf"),
    ) -> Sequence[VerificationSubproblem]:
        assert self.optimize_alpha
        start_time = time.time()
        batch_size = query_coef.shape[0]

        time_remaining = timeout - (time.time() - start_time)
        if self.optimize_prima:
            (
                improved_lbs,
                improved_ubs,
                optimized_parameters_batch,
                intermediate_bounds_batch,
                updated_prima_coefficients,
            ) = self._bound_minimum_optimizing_alpha_prima(
                query_coef,
                network,
                input_lb,
                input_ub,
                subproblem_batch.parameters_by_starting_layer,
                subproblem_batch.split_constraints,
                subproblem_batch.intermediate_layer_bounds_to_be_kept_fixed,
                subproblem_batch.intermediate_bounds,
                subproblem_batch.prima_coefficients,
                early_stopping_threshold,
                time_remaining,
            )
        else:
            (
                improved_lbs,
                improved_ubs,
                optimized_parameters_batch,
                intermediate_bounds_batch,
            ) = self._bound_minimum_optimizing_alpha(
                query_coef,
                network,
                input_lb,
                input_ub,
                subproblem_batch.split_constraints,
                subproblem_batch.intermediate_layer_bounds_to_be_kept_fixed,
                subproblem_batch.intermediate_bounds,
                subproblem_batch.parameters_by_starting_layer,
                early_stopping_threshold,
                time_remaining,
            )
            updated_prima_coefficients = {}
        print("improved lower bounds:", improved_lbs)

        self._update_best_intermediate_bounds(
            intermediate_bounds_batch,
            subproblem_batch.intermediate_bounds,
        )
        split_constraints_after_bounding = unbatch_layer_property(
            subproblem_batch.split_constraints, batch_size
        )
        intermediate_bounds = unbatch_layer_bounds(
            intermediate_bounds_batch, batch_size
        )
        optimized_parameters = unbatch_parameters(
            optimized_parameters_batch, batch_size
        )
        prima_coefficients = unbatch_prima_coefficients(
            updated_prima_coefficients, batch_size
        )
        invalid_bounds_mask_in_batch = self._get_infeasibility_mask_from(
            intermediate_bounds_batch
        )
        return [
            VerificationSubproblem(
                improved_lbs[i],
                improved_ubs[i],
                split_constraints_after_bounding[i],
                [],
                intermediate_bounds[i],
                optimized_parameters[i],
                prima_coefficients[i],
                invalid_bounds_mask_in_batch[i],
                [subproblem_batch.number_of_nodes_split[i]],
            )
            for i in range(batch_size)
        ]

    def _can_stop_early(
        self,
        lower_bounds: Sequence[float],
        upper_bounds: Sequence[float],
        early_stopping_threshold: float,
        infesibility_mask: Sequence[bool],
    ) -> bool:
        counter_example_found = any_smaller(
            upper_bounds, early_stopping_threshold
        ) and early_stopping_threshold != float("inf")
        verified_mask = (
            lower_bound > early_stopping_threshold for lower_bound in lower_bounds
        )
        verified_or_infeasible = (
            verified or infeasible
            for (verified, infeasible) in zip(verified_mask, infesibility_mask)
        )
        return counter_example_found or all(verified_or_infeasible)

    @torch.no_grad()
    def bound_minimum_with_deep_poly(
        self,
        query_coef: Tensor,
        network: AbstractNetwork,
        input_lb: Tensor,
        input_ub: Tensor,
        split_constraints: Optional[Dict[int, Tensor]] = None,
        intermediate_layers_to_be_kept_fixed: Optional[Sequence[int]] = None,
        best_intermediate_bounds_so_far: Optional[
            OrderedDict[int, Tuple[Tensor, Tensor]]
        ] = None,
        carried_over_optimizable_parameters: Optional[
            Dict[int, Dict[str, Dict[int, Tensor]]]
        ] = None,
        precomputed_prima_coefficients: Optional[
            Dict[int, Tuple[Tensor, Tensor, Tensor]]
        ] = None,
    ) -> Tuple[Sequence[float], Sequence[float]]:
        abstract_shape = MN_BaB_Shape(
            lb_coef=query_coef,
            ub_coef=query_coef,
            carried_over_optimizable_parameters=carried_over_optimizable_parameters,
            split_constraints=split_constraints,
            prima_coefficients=precomputed_prima_coefficients,
        )

        abstract_shape = network.get_mn_bab_shape(
            input_lb,
            input_ub,
            abstract_shape,
            intermediate_layers_to_be_kept_fixed,
            best_intermediate_bounds_so_far,
            use_dependence_sets=self.use_dependence_sets,
            use_early_termination=self.use_early_termination,
        )
        output_lbs, __ = abstract_shape.concretize(input_lb, input_ub)
        ubs_of_minimum = MNBabOptimizer._get_upper_bound_of_minimum(
            query_coef, abstract_shape, network, input_lb, input_ub
        )
        assert (
            (output_lbs.squeeze() <= torch.Tensor(ubs_of_minimum).to(output_lbs.device))
            | (output_lbs.squeeze() >= 0)
        ).all(), f"output_lb: {output_lbs}; output_ub_min: {ubs_of_minimum}"
        return output_lbs.flatten().tolist(), ubs_of_minimum

    def _bound_minimum_optimizing_alpha(
        self,
        query_coef: Tensor,
        network: AbstractNetwork,
        input_lb: Tensor,
        input_ub: Tensor,
        split_constraints: Optional[Dict[int, Tensor]] = None,
        intermediate_layers_to_be_kept_fixed: Optional[Sequence[int]] = None,
        best_intermediate_bounds_so_far: Optional[
            OrderedDict[int, Tuple[Tensor, Tensor]]
        ] = None,
        carried_over_optimizable_parameters: Optional[
            Dict[int, Dict[str, Dict[int, Tensor]]]
        ] = None,
        early_stopping_threshold: float = float("inf"),
        timeout: float = float("inf"),
    ) -> Tuple[
        Sequence[float],
        Sequence[float],
        Dict[int, Dict[str, Dict[int, Tensor]]],
        OrderedDict[int, Tuple[Tensor, Tensor]],
    ]:
        start_time = time.time()
        if intermediate_layers_to_be_kept_fixed is None:
            intermediate_layers_to_be_kept_fixed = []
        if carried_over_optimizable_parameters is None:
            carried_over_optimizable_parameters = {}
        abstract_shape = MN_BaB_Shape(
            lb_coef=query_coef,
            ub_coef=query_coef,
            carried_over_optimizable_parameters=carried_over_optimizable_parameters,
            split_constraints=split_constraints,
        )
        abstract_shape = network.get_mn_bab_shape(
            input_lb,
            input_ub,
            abstract_shape,
            intermediate_layers_to_be_kept_fixed,
            best_intermediate_bounds_so_far,
            use_dependence_sets=self.use_dependence_sets,
            use_early_termination=self.use_early_termination,
        )
        assert abstract_shape.carried_over_optimizable_parameters is not None

        time_remaining = timeout - (time.time() - start_time)
        return self._optimize_parameters(
            query_coef,
            network,
            input_lb,
            input_ub,
            abstract_shape,
            intermediate_layers_to_be_kept_fixed,
            best_intermediate_bounds_so_far,
            self.alpha_opt_iterations,
            early_stopping_threshold,
            time_remaining,
        )

    def _bound_minimum_optimizing_alpha_prima(
        self,
        query_coef: Tensor,
        network: AbstractNetwork,
        input_lb: Tensor,
        input_ub: Tensor,
        carried_over_optimizable_parameters: Dict[int, Dict[str, Dict[int, Tensor]]],
        split_constraints: Optional[Dict[int, Tensor]] = None,
        intermediate_layers_to_be_kept_fixed: Optional[Sequence[int]] = None,
        best_intermediate_bounds_so_far: Optional[
            OrderedDict[int, Tuple[Tensor, Tensor]]
        ] = None,
        precomputed_prima_coefficients: Optional[
            Dict[int, Tuple[Tensor, Tensor, Tensor]]
        ] = None,
        early_stopping_threshold: float = float("inf"),
        timeout: float = float("inf"),
    ) -> Tuple[
        Sequence[float],
        Sequence[float],
        Dict[int, Dict[str, Dict[int, Tensor]]],
        OrderedDict[int, Tuple[Tensor, Tensor]],
        Dict[int, Tuple[Tensor, Tensor, Tensor]],
    ]:
        start_time = time.time()
        if intermediate_layers_to_be_kept_fixed is None:
            intermediate_layers_to_be_kept_fixed = []
        if precomputed_prima_coefficients is None:
            precomputed_prima_coefficients = {}
        abstract_shape = MN_BaB_Shape(
            lb_coef=query_coef,
            ub_coef=query_coef,
            carried_over_optimizable_parameters=carried_over_optimizable_parameters,
            prima_coefficients=precomputed_prima_coefficients,
            prima_hyperparamters=self.prima_hyperparamters,
            split_constraints=split_constraints,
        )
        layer_ids_for_which_to_compute_prima_constraints = (
            network.get_activation_layer_ids()
        )
        abstract_shape = network.get_mn_bab_shape(
            input_lb,
            input_ub,
            abstract_shape,
            intermediate_layers_to_be_kept_fixed,
            best_intermediate_bounds_so_far,
            layer_ids_for_which_to_compute_prima_constraints,
            use_dependence_sets=self.use_dependence_sets,
            use_early_termination=self.use_early_termination,
        )
        assert abstract_shape.carried_over_optimizable_parameters is not None

        time_remaining = timeout - (time.time() - start_time)
        (
            best_lbs,
            best_ubs,
            best_parameters,
            best_intermediate_bounds,
        ) = self._optimize_parameters(
            query_coef,
            network,
            input_lb,
            input_ub,
            abstract_shape,
            intermediate_layers_to_be_kept_fixed,
            best_intermediate_bounds_so_far,
            self.prima_opt_iterations,
            early_stopping_threshold,
            time_remaining,
        )
        return (
            best_lbs,
            best_ubs,
            best_parameters,
            best_intermediate_bounds,
            abstract_shape.prima_coefficients,
        )

    def _optimize_parameters(
        self,
        query_coef: Tensor,
        network: AbstractNetwork,
        input_lb: Tensor,
        input_ub: Tensor,
        abstract_shape: MN_BaB_Shape,
        intermediate_layers_to_be_kept_fixed: Sequence[int],
        best_intermediate_bounds_so_far: Optional[
            OrderedDict[int, Tuple[Tensor, Tensor]]
        ],
        optimization_iterations: int,
        early_stopping_threshold: float,
        timeout: float,
    ) -> Tuple[
        Sequence[float],
        Sequence[float],
        Dict[int, Dict[str, Dict[int, Tensor]]],
        OrderedDict[int, Tuple[Tensor, Tensor]],
    ]:
        start_time = time.time()
        assert abstract_shape.carried_over_optimizable_parameters
        all_alpha_parameters = []
        all_beta_parameters = []
        all_prima_parameters = []
        for (
            optimizable_params
        ) in abstract_shape.carried_over_optimizable_parameters.values():
            for param_key in optimizable_params.keys():
                if "alpha" in param_key:
                    all_alpha_parameters += list(optimizable_params[param_key].values())
                elif "beta" in param_key:
                    all_beta_parameters += list(optimizable_params[param_key].values())
                elif "prima" in param_key:
                    all_prima_parameters += list(optimizable_params[param_key].values())
                else:
                    raise RuntimeError(
                        "Unknown optimizable parameter {}".format(param_key)
                    )

        parameters_to_optimize = [
            {"params": all_alpha_parameters, "lr": self.alpha_lr},
            {"params": all_beta_parameters, "lr": self.beta_lr},
            {"params": all_prima_parameters, "lr": self.prima_lr},
        ]

        optimizer = optim.Adam(parameters_to_optimize)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            [
                self.peak_lr_scaling_factor * self.alpha_lr,
                self.peak_lr_scaling_factor * self.beta_lr,
                self.peak_lr_scaling_factor * self.prima_lr,
            ],
            optimization_iterations,
            final_div_factor=self.final_lr_div_factor,
        )

        query_coef_copy = query_coef.clone()
        abstract_shape.update_bounds(query_coef_copy, query_coef_copy)

        best_lower_bounds = torch.tensor(
            [-float("inf")] * abstract_shape.batch_size, device=abstract_shape.device
        )
        best_upper_bounds = torch.tensor([float("inf")] * abstract_shape.batch_size)
        if (best_intermediate_bounds_so_far is not None) and (
            best_intermediate_bounds_so_far
        ):
            best_intermediate_bounds = best_intermediate_bounds_so_far
            self._update_best_intermediate_bounds(
                best_intermediate_bounds,
                network.get_current_intermediate_bounds(),
            )
        else:
            best_intermediate_bounds = network.get_current_intermediate_bounds()

        best_parameters = abstract_shape.carried_over_optimizable_parameters
        for i in range(optimization_iterations):
            if time.time() - start_time > timeout:
                raise VerificationTimeoutException()
            query_coef_copy = query_coef.clone()
            abstract_shape.update_bounds(query_coef_copy, query_coef_copy)
            abstract_shape = network.get_mn_bab_shape(
                input_lb,
                input_ub,
                abstract_shape,
                intermediate_layers_to_be_kept_fixed,
                best_intermediate_bounds,
                use_dependence_sets=self.use_dependence_sets,
                use_early_termination=self.use_early_termination,
            )
            output_lbs, __ = abstract_shape.concretize(input_lb, input_ub)
            assert abstract_shape.carried_over_optimizable_parameters is not None

            output_lbs = output_lbs.flatten()
            upper_bounds = torch.tensor(
                MNBabOptimizer._get_upper_bound_of_minimum(
                    query_coef, abstract_shape, network, input_lb, input_ub
                )
            )
            improvement_mask = output_lbs >= best_lower_bounds
            if any(improvement_mask):
                best_parameters = self._update_best_parameters(
                    best_parameters,
                    abstract_shape.carried_over_optimizable_parameters,
                    improvement_mask,
                )
            self._update_best_intermediate_bounds(
                best_intermediate_bounds,
                network.get_current_intermediate_bounds(),
            )

            best_lower_bounds = torch.maximum(output_lbs, best_lower_bounds).detach()
            best_upper_bounds = torch.minimum(upper_bounds, best_upper_bounds)

            if self._can_stop_early(
                best_lower_bounds,
                best_upper_bounds,
                early_stopping_threshold,
                abstract_shape.invalid_bounds_mask_in_batch,
            ):
                break

            not_yet_verified_lbs = output_lbs[output_lbs < early_stopping_threshold]
            loss = -not_yet_verified_lbs.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if all(
                self._all_gradients_zero_mask(
                    optimizer, abstract_shape.batch_size, abstract_shape.device
                )
            ):
                break

            for alpha_parameters in all_alpha_parameters:
                alpha_parameters.data = torch.clamp(alpha_parameters.data, 0.0, 1.0)

            for beta_parameters in all_beta_parameters:
                beta_parameters.data = torch.clamp(beta_parameters.data, min=0.0)

            for prima_parameters in all_prima_parameters:
                prima_parameters.data = torch.clamp(prima_parameters.data, min=0.0)

        return (
            best_lower_bounds.tolist(),
            best_upper_bounds.tolist(),
            best_parameters,
            best_intermediate_bounds,
        )

    @staticmethod
    def _get_upper_bound_of_minimum(
        query_coef: Tensor,
        abstract_shape: MN_BaB_Shape,
        network: AbstractNetwork,
        input_lb: Tensor,
        input_ub: Tensor,
    ) -> Sequence[float]:
        upper_bound_input = abstract_shape.get_input_corresponding_to_lower_bound(
            input_lb, input_ub
        )
        output = network(upper_bound_input)
        return torch.einsum(
            "bij, bij -> b", output.view_as(query_coef), query_coef
        ).tolist()

    def _update_best_parameters(
        self,
        prev_best_parameters: Dict[int, Dict[str, Dict[int, Tensor]]],
        new_parameters: Dict[int, Dict[str, Dict[int, Tensor]]],
        improvement_mask: Tensor,
    ) -> Dict[int, Dict[str, Dict[int, Tensor]]]:
        copied_parameters: Dict[int, Dict[str, Dict[int, Tensor]]] = {}
        for starting_layer_id, optimizable_parameters in prev_best_parameters.items():
            copied_parameters[starting_layer_id] = {}
            for param_key, parameters_per_layer in optimizable_parameters.items():
                copied_parameters[starting_layer_id][param_key] = {}
                for layer_id, layer_parameters in parameters_per_layer.items():
                    improvement_mask_of_appropriate_shape = improvement_mask.view(
                        improvement_mask.shape[0],
                        *([1] * (len(layer_parameters.shape) - 1)),
                    )
                    copied_parameters[starting_layer_id][param_key][layer_id] = (
                        torch.where(
                            improvement_mask_of_appropriate_shape,
                            new_parameters[starting_layer_id][param_key][layer_id],
                            layer_parameters,
                        )
                        .clone()
                        .detach()
                    )
        return copied_parameters

    def _update_best_intermediate_bounds(
        self,
        best_intermediate_bounds: OrderedDict[int, Tuple[Tensor, Tensor]],
        current_intermediate_bounds: OrderedDict[int, Tuple[Tensor, Tensor]],
    ) -> None:
        for layer_id in best_intermediate_bounds:
            if layer_id in current_intermediate_bounds:
                best_intermediate_bounds[layer_id] = (
                    torch.maximum(
                        best_intermediate_bounds[layer_id][0],
                        current_intermediate_bounds[layer_id][0],
                    )
                    .clone()
                    .detach(),
                    torch.minimum(
                        best_intermediate_bounds[layer_id][1],
                        current_intermediate_bounds[layer_id][1],
                    )
                    .clone()
                    .detach(),
                )

    def _get_infeasibility_mask_from(
        self,
        intermediate_bounds: OrderedDict[int, Tuple[Tensor, Tensor]],
    ) -> Sequence[bool]:
        batch_size = list(intermediate_bounds.values())[0][0].shape[0]
        return [
            any(
                (
                    intermediate_bounds[layer_id][0][batch_index]
                    > intermediate_bounds[layer_id][1][batch_index]
                    + INFEASIBILITY_CHECK_TOLERANCE
                )
                .any()
                .item()
                for layer_id in intermediate_bounds
            )
            for batch_index in range(batch_size)
        ]

    def _all_gradients_zero_mask(
        self, optimizer: Optimizer, batch_size: int, device: torch.device
    ) -> Sequence[bool]:
        gradients_zero_mask = torch.tensor(
            [True for __ in range(batch_size)], device=device
        )
        for param_group in optimizer.param_groups:
            parameters = param_group["params"]
            for parameter_batch in parameters:
                if parameter_batch.grad is None:
                    continue
                assert parameter_batch.shape[0] == batch_size
                flattened_parameter_batch_grad = parameter_batch.grad.view(
                    parameter_batch.shape[0], -1
                )
                parameter_gradient_zero_mask = torch.all(
                    flattened_parameter_batch_grad == 0, dim=1
                )
                gradients_zero_mask = torch.logical_and(
                    gradients_zero_mask, parameter_gradient_zero_mask
                )
                if not gradients_zero_mask.any():
                    return gradients_zero_mask.tolist()
        return gradients_zero_mask.tolist()
