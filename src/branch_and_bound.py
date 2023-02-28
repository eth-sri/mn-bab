import time
from typing import Any, Dict, Optional, OrderedDict, Sequence, Tuple

from comet_ml import Experiment  # type: ignore[import]

import torch
from torch import Tensor
from tqdm import tqdm  # type: ignore[import]

from src.abstract_layers.abstract_network import AbstractNetwork
from src.exceptions.verification_timeout import VerificationTimeoutException
from src.mn_bab_optimizer import MNBabOptimizer
from src.state.tags import LayerTag, NodeTag
from src.utilities.batching import batch_subproblems, unbatch_subproblems
from src.utilities.branching import make_split_index_finder
from src.utilities.config import BacksubstitutionConfig, BranchAndBoundConfig
from src.verification_subproblem import (
    ReadonlyVerificationSubproblem,
    VerificationSubproblem,
)
from src.verification_subproblem_queue import VerificationSubproblemQueue


class BranchAndBound:
    def __init__(
        self,
        optimizer: MNBabOptimizer,
        config: BranchAndBoundConfig,
        backsubstitution_config: BacksubstitutionConfig,
        device: torch.device,
    ):
        self.optimizer = optimizer
        self.config = config
        self.backsubstitution_config = backsubstitution_config
        self.logger = BranchAndBoundLogger()
        self.device = device
        self.cpu_device = torch.device("cpu")

    def bound_minimum_with_branch_and_bound(
        self,
        property_id: str,
        query_coef: Tensor,
        network: AbstractNetwork,
        input_lb: Tensor,
        input_ub: Tensor,
        early_stopping_threshold: Optional[float] = None,
        timeout: float = float("inf"),
        initial_bounds: Optional[OrderedDict[LayerTag, Tuple[Tensor, Tensor]]] = None,
    ) -> Tuple[float, float, Optional[Tensor]]:
        """
        Finds (lower_bound, upper_bound) satisfying
        lower_bound <= min_x query*network(x) <= upper_bound
        using a branch and bound approach.
        """
        assert (
            query_coef.shape[0] == 1
        ), "Expected single query."  # TODO: maybe we can batch multiple different queries?
        self.logger.init_property(property_id)

        start_time = time.time()

        initial_bounded_problem, ub_inputs = self.optimizer.bound_root_subproblem(
            input_lb,
            input_ub,
            network,
            query_coef,
            early_stopping_threshold,
            timeout,
            self.device,
            initial_bounds=initial_bounds,
        )
        network.activation_layer_bounds_to_optim_layer_bounds()
        self.logger.add_bounded_subproblems(property_id, 1)
        initial_bounded_problem.move_to(self.cpu_device)

        # Early stop for fast benchmarking
        # if initial_bounded_problem.lower_bound < -7:
        #    return initial_bounded_problem.lower_bound

        if not self.config.run_BaB:
            return (
                initial_bounded_problem.lower_bound,
                initial_bounded_problem.upper_bound,
                None,
            )

        if (
            early_stopping_threshold is not None
            and early_stopping_threshold < initial_bounded_problem.lower_bound
        ):
            # property already shown for initial_bounded_problem
            self.logger.verified_subproblem(property_id, initial_bounded_problem)
            self.logger.verified(property_id, time.time() - start_time, lower_bound=0.0)
            return (
                initial_bounded_problem.lower_bound,
                initial_bounded_problem.upper_bound,
                None,
            )

        if (
            early_stopping_threshold is not None
            and initial_bounded_problem.upper_bound < early_stopping_threshold
        ):
            # primal solution below threshold found ==> counterexample found
            self.logger.counterexample(
                property_id, time.time() - start_time, None, early_stopping_threshold
            )
            return (
                initial_bounded_problem.lower_bound,
                initial_bounded_problem.upper_bound,
                None if ub_inputs is None else ub_inputs.view(-1, *input_lb.shape[1:]),
            )

        subproblems_to_be_refined = VerificationSubproblemQueue(
            initial_bounded_problem,
            self.config.batch_sizes,
        )

        split_index_finder = make_split_index_finder(
            network,
            self.backsubstitution_config,  # TODO: should this backsubstitution config have prima hyperparameters? (does it even matter for branching score computations?)
            query_coef,
            initial_bounded_problem,
            self.config.branching_config,
            # (the following parameters are only used for filtered smart branching)
            input_lb,
            input_ub,
            self.config.batch_sizes,
            self.config.recompute_intermediate_bounds_after_branching,
            self.optimizer,
        )

        verified_lb = float("inf")  # smallest lower bound of a verified subproblem
        global_lb = initial_bounded_problem.lower_bound  # smallest lower bound overall
        global_ub = (
            initial_bounded_problem.upper_bound
        )  # smallest upper bound on minimum
        last_time = time.time()
        with tqdm(total=timeout - time.time(), bar_format="{l_bar}{bar}") as tq:
            while True:  # terminates via return from function
                next_subproblem = subproblems_to_be_refined.peek()
                assert next_subproblem is not None

                unverified_lb = next_subproblem.lower_bound
                global_lb = unverified_lb  # unverified_lb < verified_lb

                if next_subproblem.is_fully_split:
                    # can't improve further
                    self.logger.fail(property_id, time.time() - start_time)
                    return (global_lb, global_ub, ub_inputs)

                if (
                    early_stopping_threshold is None and global_ub - global_lb <= 1e-5
                ):  # round-off epsilon
                    self.logger.verified(
                        property_id, time.time() - start_time, lower_bound=0.0
                    )  # TODO: why 0.0?
                    return (global_lb, global_ub, ub_inputs)

                # print()
                # print("global_lb:", global_lb)
                # print("global_ub:", global_ub)
                # print("queue length:", len(subproblems_to_be_refined))
                time_remaining = timeout + start_time - time.time()
                tq.set_description_str(
                    f"Global LB: {global_lb:.5f} | Global UB: {global_ub:.4f} | Queue Length: {len(subproblems_to_be_refined)} | Subproblems Considered: {self.logger.info['number_of_subproblems_bounded'][property_id]} | TR: {time_remaining:.2f}"
                )
                curr_time = time.time()
                tq.update(curr_time - last_time)
                last_time = curr_time
                self.logger.lower_bound(property_id, global_lb)

                def find_node_to_split(
                    subproblem: ReadonlyVerificationSubproblem,
                ) -> NodeTag:
                    node_to_split = split_index_finder.find_node_to_split(subproblem)

                    layer_id_to_split = node_to_split.layer
                    layer_index_to_split = subproblems_to_be_refined._layer_id_to_index[
                        layer_id_to_split
                    ]
                    # print(f"splitting on layer {layer_index_to_split}")
                    self.logger.split_layer(layer_index_to_split)
                    return node_to_split

                next_subproblems = subproblems_to_be_refined.pop(
                    find_node_to_split,
                    self.config.recompute_intermediate_bounds_after_branching,
                    network,
                )
                self.logger.add_to_number_of_splits(
                    property_id, len(next_subproblems) // 2
                )

                try:
                    bounded_subproblems, ub_inputs = self._bound_minimum_in_batch(
                        next_subproblems,
                        query_coef,
                        network,
                        input_lb,
                        input_ub,
                        early_stopping_threshold,
                        time_remaining,
                    )
                    self.logger.add_bounded_subproblems(
                        property_id, len(bounded_subproblems)
                    )
                except VerificationTimeoutException as e:
                    queue_length = len(subproblems_to_be_refined)
                    self.logger.timeout(property_id, global_lb, global_ub, queue_length)
                    e.best_lb = global_lb
                    e.best_ub = global_ub
                    raise e

                min_upper_bound_in_batch = min(
                    bounded_subproblem.upper_bound
                    for bounded_subproblem in bounded_subproblems
                )
                global_ub = min(min_upper_bound_in_batch, global_ub)
                counterexample_found = (
                    early_stopping_threshold is not None
                    and global_ub < early_stopping_threshold
                )
                if counterexample_found:
                    self.logger.counterexample(
                        property_id,
                        time.time() - start_time,
                        bounded_subproblems,
                        early_stopping_threshold,
                    )
                    return (global_lb, global_ub, ub_inputs)

                for bounded_subproblem in bounded_subproblems:
                    if (
                        bounded_subproblem.is_infeasible
                        or early_stopping_threshold is not None
                        and early_stopping_threshold < bounded_subproblem.lower_bound
                        or global_ub < bounded_subproblem.lower_bound
                    ):
                        self.logger.verified_subproblem(property_id, bounded_subproblem)
                        verified_lb = min(verified_lb, bounded_subproblem.lower_bound)
                    else:
                        subproblems_to_be_refined.insert_sorted(bounded_subproblem)

                if (
                    subproblems_to_be_refined.empty
                ):  # exhausted queue, everything verified
                    global_lb = verified_lb
                    self.logger.verified(
                        property_id, time.time() - start_time, lower_bound=0.0
                    )  # TODO: why 0.0?
                    return (global_lb, global_ub, ub_inputs)

    def _bound_minimum_in_batch(
        self,
        subproblems: Sequence[ReadonlyVerificationSubproblem],
        query_coef: Tensor,
        network: AbstractNetwork,
        input_lb: Tensor,
        input_ub: Tensor,
        early_stopping_threshold: Optional[float],
        timeout: float,
    ) -> Tuple[Sequence[VerificationSubproblem], Tensor]:
        assert all(subproblem.device == self.cpu_device for subproblem in subproblems)
        subproblem_batch = batch_subproblems(
            [subproblem.deep_copy_to(self.device) for subproblem in subproblems],
            reuse_single_subproblem=True,
        )
        subproblem_state_batch = subproblem_batch.subproblem_state
        query_coef = query_coef.to(self.device)
        batch_size = len(subproblems)
        assert subproblem_state_batch.batch_size == batch_size
        batch_repeats = batch_size, *([1] * (len(query_coef.shape) - 1))

        (
            improved_lbs,
            improved_ubs,
            ub_inputs,
        ) = self.optimizer.improve_subproblem_batch_bounds(
            subproblem_state_batch,  # (updated in-place)
            input_lb,
            input_ub,
            network,
            query_coef.repeat(batch_repeats),
            early_stopping_threshold,
            self.config.recompute_intermediate_bounds_after_branching,
            timeout,
        )

        result = unbatch_subproblems(
            subproblem_state_batch,
            improved_lbs,
            improved_ubs,
            reset_intermediate_layer_bounds_to_be_kept_fixed=True,
        )
        for subproblem in result:
            subproblem.move_to(self.cpu_device)
        return (result, ub_inputs)

    def log_info(self, experiment_logger: Experiment) -> None:
        self.logger.log_info(experiment_logger)


class BranchAndBoundLogger:
    def __init__(self) -> None:
        self.info: Dict[str, Any] = {
            "number_of_subproblems_bounded": {},
            "verification_time": {},
            "total_number_of_splits": {},
            "number_of_splits_per_layer": {},
            "max_split_depth_needed_to_verify": {},
            "split_depth_needed_for_counterexample": {},
            "lower_bound_at_timeout": {},
            "upper_bound_at_timeout": {},
            "queue_length_at_timeout": {},
            "n_subproblems_explored_to_reach_lower_bound": {},
        }

    def init_property(self, property_id: str) -> None:
        self.info["number_of_subproblems_bounded"][property_id] = 0
        self.info["total_number_of_splits"][property_id] = 0
        self.info["n_subproblems_explored_to_reach_lower_bound"][property_id] = []
        self.info["max_split_depth_needed_to_verify"][property_id] = 0

    def add_bounded_subproblems(
        self, property_id: str, num_bounded_subproblems: int
    ) -> None:
        self.info["number_of_subproblems_bounded"][
            property_id
        ] += num_bounded_subproblems

    def lower_bound(self, property_id: str, lower_bound: float) -> None:
        self.info["n_subproblems_explored_to_reach_lower_bound"][property_id].append(
            (
                lower_bound,
                self.info["number_of_subproblems_bounded"][property_id],
            )
        )

    def split_layer(self, layer_index_to_split: int) -> None:
        n_splits = self.info["number_of_splits_per_layer"].setdefault(
            layer_index_to_split, 0
        )
        self.info["number_of_splits_per_layer"][layer_index_to_split] = n_splits + 1

    def add_to_number_of_splits(
        self, property_id: str, number_of_splits: int
    ) -> None:  # TODO: couldn't this just be counted in split_layer?
        self.info["total_number_of_splits"][property_id] += number_of_splits

    def verified_subproblem(
        self, property_id: str, bounded_subproblem: ReadonlyVerificationSubproblem
    ) -> None:
        if bounded_subproblem is not None:
            split_depth = (
                0
                if bounded_subproblem.subproblem_state.constraints.split_state is None
                else max(
                    bounded_subproblem.subproblem_state.constraints.split_state.number_of_nodes_split
                )
            )  # TODO: ugly
        else:
            split_depth = 0
        cur_max_split_depth = self.info["max_split_depth_needed_to_verify"][property_id]
        self.info["max_split_depth_needed_to_verify"][property_id] = max(
            cur_max_split_depth, split_depth
        )

    def verified(
        self, property_id: str, verification_time: float, lower_bound: float
    ) -> None:
        self.info["verification_time"][property_id] = verification_time
        self.lower_bound(property_id, lower_bound)

    def counterexample(
        self,
        property_id: str,
        verification_time: float,
        bounded_subproblems: Optional[Sequence[ReadonlyVerificationSubproblem]],
        early_stopping_threshold: Optional[float],
    ) -> None:
        print("counterexample found, stopping")

        if bounded_subproblems is not None:
            assert all(
                subproblem.subproblem_state.constraints.split_state is not None
                for subproblem in bounded_subproblems
            )
            depth_needed_for_counterexample = min(
                n
                for subproblem in bounded_subproblems
                for n in subproblem.subproblem_state.constraints.split_state.number_of_nodes_split  # type: ignore[union-attr] # mypy can't see split_state is not None
                if early_stopping_threshold is not None
                and subproblem.upper_bound < early_stopping_threshold
            )
        else:
            depth_needed_for_counterexample = 0

        self.info["split_depth_needed_for_counterexample"][
            property_id
        ] = depth_needed_for_counterexample
        del self.info["max_split_depth_needed_to_verify"][property_id]
        self.info["verification_time"][property_id] = verification_time

    def fail(self, property_id: str, verification_time: float) -> None:
        print("failed to verify")
        self.info["verification_time"][property_id] = verification_time

    def timeout(
        self,
        property_id: str,
        lower_bound: float,
        upper_bound: float,
        queue_length: int,
    ) -> None:
        self.info["lower_bound_at_timeout"][property_id] = lower_bound
        self.info["upper_bound_at_timeout"][property_id] = upper_bound
        self.info["queue_length_at_timeout"][property_id] = queue_length
        self.info["max_split_depth_needed_to_verify"][property_id] = float("inf")
        self.info["number_of_subproblems_bounded"][property_id] = float("inf")
        self.info["verification_time"][property_id] = float("inf")

    def log_info(self, experiment_logger: Experiment) -> None:
        experiment_logger.log_asset_data(self.info, name="log_info.json")
        non_zero_number_of_splits = [
            n_splits
            for n_splits in self.info["total_number_of_splits"].values()
            if n_splits != 0
        ]
        experiment_logger.log_histogram_3d(
            non_zero_number_of_splits,
            step=1,
            name="total_number_of_splits_if_non_zero",
        )
        experiment_logger.log_histogram_3d(
            list(self.info["total_number_of_splits"].values()),
            step=1,
            name="total_number_of_splits",
        )
        experiment_logger.log_histogram_3d(
            list(self.info["split_depth_needed_for_counterexample"].values()),
            step=1,
            name="split_depth_needed_for_counterexample",
        )
        experiment_logger.log_histogram_3d(
            [
                depth
                for depth in self.info["max_split_depth_needed_to_verify"].values()
                if depth != float("inf")
            ],
            step=1,
            name="max_split_depth_needed_to_verify",
        )
        non_zero_max_depth_to_verify = [
            depth
            for depth in self.info["max_split_depth_needed_to_verify"].values()
            if depth != 0 and depth != float("inf")
        ]
        experiment_logger.log_histogram_3d(
            non_zero_max_depth_to_verify,
            step=1,
            name="max_split_depth_needed_to_verify_if_non_zero",
        )
        experiment_logger.log_histogram_3d(
            list(self.info["lower_bound_at_timeout"].values()),
            step=1,
            name="lower_bound_at_timeout",
        )
        experiment_logger.log_histogram_3d(
            list(self.info["upper_bound_at_timeout"].values()),
            step=1,
            name="upper_bound_at_timeout",
        )
        experiment_logger.log_histogram_3d(
            list(self.info["queue_length_at_timeout"].values()),
            step=1,
            name="queue_length_at_timeout",
        )
        experiment_logger.log_histogram_3d(
            self.info["number_of_splits_per_layer"],
            step=1,
            name="number_of_splits_per_layer",
        )
