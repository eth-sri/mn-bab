import time
from typing import Any, Dict, Sequence, Tuple

from comet_ml import Experiment

import torch
from torch import Tensor

from src.abstract_layers.abstract_network import AbstractNetwork
from src.exceptions.verification_timeout import VerificationTimeoutException
from src.mn_bab_optimizer import MNBabOptimizer
from src.utilities.batching import batch_layer_properties
from src.utilities.branching import (
    compute_split_cost_by_layer,
    find_index_to_split_with_babsr,
    find_index_to_split_with_filtered_smart_branching,
)
from src.verification_subproblem import VerificationSubproblem
from src.verification_subproblem_queue import VerificationSubproblemQueue


class BranchAndBound:
    def __init__(
        self,
        optimizer: MNBabOptimizer,
        batch_sizes: Sequence[int],
        branching_config: Dict[str, Any],
        recompute_intermediate_bounds_after_branching: bool,
        device: torch.device = torch.device("cpu"),
    ):
        self.optimizer = optimizer
        self.batch_sizes = batch_sizes
        self.branching_config = branching_config
        self.recompute_intermediate_bounds_after_branching = (
            recompute_intermediate_bounds_after_branching
        )
        self.logging_info: Dict[str, Any] = {
            "number_of_subproblems_bounded": {},
            "verfication_time": {},
            "total_number_of_splits": {},
            "number_of_splits_per_layer": {},
            "max_split_depth_needed_to_verify": {},
            "split_depth_needed_for_counterexample": {},
            "lower_bound_at_timeout": {},
            "upper_bound_at_timeout": {},
            "queue_length_at_timeout": {},
            "n_subproblems_explored_to_reach_lower_bound": {},
        }
        self.device = device

    def lower_bound_property_with_branch_and_bound(
        self,
        property_id: str,
        query_coef: Tensor,
        network: AbstractNetwork,
        input_lb: Tensor,
        input_ub: Tensor,
        early_stopping_threshold: float = float("inf"),
        timeout: float = float("inf"),
    ) -> float:
        self.logging_info["total_number_of_splits"][property_id] = 0
        self.logging_info["max_split_depth_needed_to_verify"][property_id] = 0
        self.logging_info["n_subproblems_explored_to_reach_lower_bound"][
            property_id
        ] = []
        start_time = time.time()
        is_early_stopping_enabled = early_stopping_threshold != float("inf")
        initial_bounded_problem = self.optimizer.bound_root_subproblem(
            query_coef,
            network,
            input_lb,
            input_ub,
            early_stopping_threshold,
            timeout,
            self.device,
        )

        self.logging_info["number_of_subproblems_bounded"][property_id] = 1
        if initial_bounded_problem.lower_bound > early_stopping_threshold:
            self.logging_info["verfication_time"][property_id] = (
                time.time() - start_time
            )
            self.logging_info["n_subproblems_explored_to_reach_lower_bound"][
                property_id
            ].append(
                (
                    0.0,
                    self.logging_info["number_of_subproblems_bounded"][property_id],
                )
            )
            return initial_bounded_problem.lower_bound
        if (
            initial_bounded_problem.upper_bound < early_stopping_threshold
            and is_early_stopping_enabled
        ):
            print("counterexample found, stopping")
            self.logging_info["split_depth_needed_for_counterexample"][property_id] = 0
            del self.logging_info["max_split_depth_needed_to_verify"][property_id]
            self.logging_info["verfication_time"][property_id] = (
                time.time() - start_time
            )
            return initial_bounded_problem.upper_bound

        subproblems_to_be_refined = VerificationSubproblemQueue(
            initial_bounded_problem,
            self.batch_sizes,
        )

        split_cost_by_layer = None
        if self.branching_config["use_cost_adjusted_scores"]:
            split_cost_by_layer = compute_split_cost_by_layer(
                network,
                initial_bounded_problem.prima_coefficients,
                self.recompute_intermediate_bounds_after_branching,
            )

        global_lb = initial_bounded_problem.lower_bound
        global_ub = initial_bounded_problem.upper_bound
        while global_ub - global_lb > 1e-5:
            print()
            print("global_lb:", global_lb)
            print("global_ub:", global_ub)
            print("queue length:", len(subproblems_to_be_refined))
            self.logging_info["n_subproblems_explored_to_reach_lower_bound"][
                property_id
            ].append(
                (
                    global_lb,
                    self.logging_info["number_of_subproblems_bounded"][property_id],
                )
            )

            def find_node_to_split(p: VerificationSubproblem) -> Tuple[int, ...]:
                if self.branching_config["method"] == "babsr":
                    node_to_split = find_index_to_split_with_babsr(
                        p,
                        network,
                        query_coef,
                        split_cost_by_layer,
                        self.branching_config["use_prima_contributions"],
                        self.branching_config["use_optimized_slopes"],
                        self.branching_config["use_beta_contributions"],
                        self.branching_config["propagation_effect_mode"],
                        self.branching_config["use_indirect_effect"],
                        self.branching_config["reduce_op"],
                        self.branching_config["use_abs"],
                        False,
                    )
                elif self.branching_config["method"] == "active_constraint_score":
                    node_to_split = find_index_to_split_with_babsr(
                        p,
                        network,
                        query_coef,
                        split_cost_by_layer,
                        True,
                        True,
                        True,
                        "none",
                        False,
                        "min",
                        False,
                        True,
                    )
                elif self.branching_config["method"] == "filtered_smart_branching":
                    node_to_split = find_index_to_split_with_filtered_smart_branching(
                        p,
                        self.optimizer,
                        network,
                        query_coef,
                        split_cost_by_layer,
                        input_lb,
                        input_ub,
                        self.branching_config["n_candidates"],
                        self.branching_config["reduce_op"],
                        self.batch_sizes,
                        self.recompute_intermediate_bounds_after_branching,
                    )
                else:
                    raise RuntimeError("Branching method misspecified.")
                layer_id_to_split = node_to_split[0]
                layer_index_to_split = subproblems_to_be_refined._layer_id_to_index[
                    layer_id_to_split
                ]
                n_splits = self.logging_info["number_of_splits_per_layer"].setdefault(
                    layer_index_to_split, 0
                )
                self.logging_info["number_of_splits_per_layer"][
                    layer_index_to_split
                ] = (n_splits + 1)
                return node_to_split

            next_subproblems = subproblems_to_be_refined.pop(
                find_node_to_split, self.recompute_intermediate_bounds_after_branching
            )
            self.logging_info["total_number_of_splits"][property_id] += (
                len(next_subproblems) / 2
            )
            time_remaining = timeout - (time.time() - start_time)
            try:
                bounded_subproblems = self._bound_minimum_in_batch(
                    next_subproblems,
                    query_coef,
                    network,
                    input_lb,
                    input_ub,
                    early_stopping_threshold,
                    time_remaining,
                )
                self.logging_info["number_of_subproblems_bounded"][property_id] += len(
                    next_subproblems
                )
            except VerificationTimeoutException:
                self.logging_info["lower_bound_at_timeout"][property_id] = global_lb
                self.logging_info["upper_bound_at_timeout"][property_id] = global_ub
                self.logging_info["queue_length_at_timeout"][property_id] = len(
                    subproblems_to_be_refined
                )
                self.logging_info["max_split_depth_needed_to_verify"][
                    property_id
                ] = float("inf")
                self.logging_info["number_of_subproblems_bounded"][property_id] = float(
                    "inf"
                )
                self.logging_info["verfication_time"][property_id] = float("inf")
                raise
            counterexample_found = (
                any(
                    [
                        bounded_subproblem.upper_bound < early_stopping_threshold
                        for bounded_subproblem in bounded_subproblems
                    ]
                )
                and is_early_stopping_enabled
            )
            if counterexample_found:
                print("counterexample found, stopping")
                depth_needed_for_counterexample = min(
                    [
                        n
                        for subproblem in bounded_subproblems
                        for n in subproblem.number_of_nodes_split
                        if subproblem.upper_bound < early_stopping_threshold
                    ]
                )
                self.logging_info["split_depth_needed_for_counterexample"][
                    property_id
                ] = depth_needed_for_counterexample
                del self.logging_info["max_split_depth_needed_to_verify"][property_id]
                global_ub = -float("inf")
                break
            else:
                min_upper_bound_in_batch = min(
                    [
                        bounded_subproblem.upper_bound
                        for bounded_subproblem in bounded_subproblems
                    ]
                )
                global_ub = min(min_upper_bound_in_batch, global_ub)
            for bounded_subproblem in bounded_subproblems:
                if (
                    bounded_subproblem.is_infeasible
                    or bounded_subproblem.lower_bound > early_stopping_threshold
                    or bounded_subproblem.lower_bound > global_ub
                ):
                    self.logging_info["max_split_depth_needed_to_verify"][
                        property_id
                    ] = max(
                        self.logging_info["max_split_depth_needed_to_verify"][
                            property_id
                        ],
                        max(bounded_subproblem.number_of_nodes_split),
                    )
                    continue
                else:
                    subproblems_to_be_refined.insert_sorted(bounded_subproblem)

            next_subproblem = subproblems_to_be_refined.peek()
            if next_subproblem is not None:
                global_lb = next_subproblem.lower_bound
                if next_subproblem.is_fully_split:
                    # can't improve further
                    global_ub = global_lb
                    break
            else:
                self.logging_info["n_subproblems_explored_to_reach_lower_bound"][
                    property_id
                ].append(
                    (
                        0.0,
                        self.logging_info["number_of_subproblems_bounded"][property_id],
                    )
                )
                break

        self.logging_info["verfication_time"][property_id] = time.time() - start_time
        return global_ub

    def _bound_minimum_in_batch(
        self,
        subproblems: Sequence[VerificationSubproblem],
        query_coef: Tensor,
        network: AbstractNetwork,
        input_lb: Tensor,
        input_ub: Tensor,
        early_stopping_threshold: float,
        timeout: float,
    ) -> Sequence[VerificationSubproblem]:
        for subproblem in subproblems:
            subproblem.to(self.device)
        subproblem_batch = batch_layer_properties(subproblems)
        query_coef = query_coef.to(self.device)
        batch_size = len(subproblems)
        batch_repeats = batch_size, *([1] * (len(query_coef.shape) - 1))
        return self.optimizer.improve_subproblem_batch_bounds(
            query_coef.repeat(batch_repeats),
            network,
            input_lb,
            input_ub,
            subproblem_batch,
            early_stopping_threshold,
            timeout,
        )

    def log_info(self, experiment_logger: Experiment) -> None:
        experiment_logger.log_asset_data(self.logging_info, name="log_info.json")
        non_zero_number_of_splits = [
            n_splits
            for n_splits in self.logging_info["total_number_of_splits"].values()
            if n_splits != 0
        ]
        experiment_logger.log_histogram_3d(
            non_zero_number_of_splits,
            step=1,
            name="total_number_of_splits_if_non_zero",
        )
        experiment_logger.log_histogram_3d(
            list(self.logging_info["total_number_of_splits"].values()),
            step=1,
            name="total_number_of_splits",
        )
        experiment_logger.log_histogram_3d(
            list(self.logging_info["split_depth_needed_for_counterexample"].values()),
            step=1,
            name="split_depth_needed_for_counterexample",
        )
        experiment_logger.log_histogram_3d(
            [
                depth
                for depth in self.logging_info[
                    "max_split_depth_needed_to_verify"
                ].values()
                if depth != float("inf")
            ],
            step=1,
            name="max_split_depth_needed_to_verify",
        )
        non_zero_max_depth_to_verify = [
            depth
            for depth in self.logging_info["max_split_depth_needed_to_verify"].values()
            if depth != 0 and depth != float("inf")
        ]
        experiment_logger.log_histogram_3d(
            non_zero_max_depth_to_verify,
            step=1,
            name="max_split_depth_needed_to_verify_if_non_zero",
        )
        experiment_logger.log_histogram_3d(
            list(self.logging_info["lower_bound_at_timeout"].values()),
            step=1,
            name="lower_bound_at_timeout",
        )
        experiment_logger.log_histogram_3d(
            list(self.logging_info["upper_bound_at_timeout"].values()),
            step=1,
            name="upper_bound_at_timeout",
        )
        experiment_logger.log_histogram_3d(
            list(self.logging_info["queue_length_at_timeout"].values()),
            step=1,
            name="queue_length_at_timeout",
        )
        experiment_logger.log_histogram_3d(
            self.logging_info["number_of_splits_per_layer"],
            step=1,
            name="number_of_splits_per_layer",
        )
