from typing import Callable, List, Optional, Sequence

import numpy as np
import torch

from src.abstract_layers.abstract_network import AbstractNetwork
from src.state.tags import NodeTag
from src.verification_subproblem import ReadonlyVerificationSubproblem


class VerificationSubproblemQueue:
    """Priority queue of VerificationSubproblems sorted by lower bound in ascending order."""

    def __init__(
        self,
        initial_subproblem: ReadonlyVerificationSubproblem,
        batch_sizes: Sequence[int],
    ) -> None:
        self._layer_id_to_index = initial_subproblem.get_layer_id_to_index()
        self._batch_sizes = batch_sizes
        self._not_yet_split_queue: List[ReadonlyVerificationSubproblem] = []
        self._queues_by_intermediate_bounds_to_keep: Sequence[
            List[ReadonlyVerificationSubproblem]
        ] = [[] for __ in range(len(batch_sizes))]
        self.global_prima_constraints = initial_subproblem.get_prima_constraints()
        self.insert_sorted(initial_subproblem)

    @property
    def empty(self) -> bool:
        return not self._not_yet_split_queue and not any(
            self._queues_by_intermediate_bounds_to_keep
        )

    def __len__(self) -> int:
        return len(self._not_yet_split_queue) + sum(
            (len(queue) // 2) for queue in self._queues_by_intermediate_bounds_to_keep
        )

    def peek(self) -> Optional[ReadonlyVerificationSubproblem]:
        if self.empty:
            return None
        minimum_lower_bound_not_yet_split = (
            self._not_yet_split_queue[0].lower_bound
            if self._not_yet_split_queue
            else torch.inf
        )
        minimum_lower_bounds_already_split = [
            subproblem_queue[0].lower_bound if subproblem_queue else torch.inf
            for subproblem_queue in self._queues_by_intermediate_bounds_to_keep
        ]
        if minimum_lower_bound_not_yet_split < min(minimum_lower_bounds_already_split):
            return self._not_yet_split_queue[0]
        else:
            index_of_minimum = int(np.argmin(minimum_lower_bounds_already_split))
            return self._queues_by_intermediate_bounds_to_keep[index_of_minimum][0]

    def pop(
        self,
        find_node_to_split: Callable[[ReadonlyVerificationSubproblem], NodeTag],
        recompute_intermediate_bounds_after_branching: bool,
        network: AbstractNetwork,
    ) -> Sequence[ReadonlyVerificationSubproblem]:
        minimum_lower_bound_not_yet_split = (
            self._not_yet_split_queue[0].lower_bound
            if self._not_yet_split_queue
            else torch.inf
        )
        minimum_lower_bounds_already_split = [
            subproblem_queue[0].lower_bound if subproblem_queue else torch.inf
            for subproblem_queue in self._queues_by_intermediate_bounds_to_keep
        ]
        if min(minimum_lower_bounds_already_split) < minimum_lower_bound_not_yet_split:
            index_of_last_intermediate_bounds_kept = int(
                np.argmin(minimum_lower_bounds_already_split)
            )
        else:
            minimum_subproblem = self._not_yet_split_queue[0]
            if not minimum_subproblem.is_fully_split:
                del self._not_yet_split_queue[0]
                node_to_split = find_node_to_split(minimum_subproblem)
                layer = network.layer_id_to_layer[node_to_split.layer]
                index_of_last_intermediate_bounds_kept = (
                    self._layer_id_to_index[node_to_split.layer]
                    if recompute_intermediate_bounds_after_branching
                    else -1
                )
                (
                    negatively_split_subproblem,
                    positively_split_subproblem,
                ) = minimum_subproblem.split(
                    node_to_split, recompute_intermediate_bounds_after_branching, layer
                )

                self._insert_sorted_into(
                    negatively_split_subproblem,
                    self._queues_by_intermediate_bounds_to_keep[
                        index_of_last_intermediate_bounds_kept
                    ],
                )
                self._insert_sorted_into(
                    positively_split_subproblem,
                    self._queues_by_intermediate_bounds_to_keep[
                        index_of_last_intermediate_bounds_kept
                    ],
                )

        while (
            len(
                self._queues_by_intermediate_bounds_to_keep[
                    index_of_last_intermediate_bounds_kept
                ]
            )
            < self._batch_sizes[index_of_last_intermediate_bounds_kept]
            and self._not_yet_split_queue
        ):
            subproblem_to_split = self._not_yet_split_queue[0]
            if subproblem_to_split.is_fully_split:
                break
            del self._not_yet_split_queue[0]
            node_to_split = find_node_to_split(subproblem_to_split)
            layer = network.layer_id_to_layer[node_to_split.layer]
            current_split_index_of_last_intermediate_bounds_kept = (
                self._layer_id_to_index[node_to_split.layer]
                if recompute_intermediate_bounds_after_branching
                else -1
            )
            (
                negatively_split_subproblem,
                positively_split_subproblem,
            ) = subproblem_to_split.split(
                node_to_split, recompute_intermediate_bounds_after_branching, layer
            )

            self._insert_sorted_into(
                negatively_split_subproblem,
                self._queues_by_intermediate_bounds_to_keep[
                    current_split_index_of_last_intermediate_bounds_kept
                ],
            )
            self._insert_sorted_into(
                positively_split_subproblem,
                self._queues_by_intermediate_bounds_to_keep[
                    current_split_index_of_last_intermediate_bounds_kept
                ],
            )
        return self._pop_from(
            index_of_last_intermediate_bounds_kept,
            self._batch_sizes[index_of_last_intermediate_bounds_kept],
        )

    def _pop_from(
        self, layer_index_to_split: int, number_of_subproblems_to_fetch: int
    ) -> Sequence[ReadonlyVerificationSubproblem]:
        max_number_of_subproblems_to_fetch = min(
            number_of_subproblems_to_fetch,
            len(self._queues_by_intermediate_bounds_to_keep[layer_index_to_split]),
        )
        subproblems_retrieved = []
        for subproblem in self._queues_by_intermediate_bounds_to_keep[
            layer_index_to_split
        ][:max_number_of_subproblems_to_fetch]:
            subproblems_retrieved.append(subproblem)
        number_of_subproblems_fetched = len(subproblems_retrieved)
        del self._queues_by_intermediate_bounds_to_keep[layer_index_to_split][
            :number_of_subproblems_fetched
        ]
        return subproblems_retrieved

    def insert_sorted(
        self,
        subproblem: ReadonlyVerificationSubproblem,
    ) -> None:
        assert (
            not subproblem.is_infeasible
        ), "Queue should only hold feasible subproblems."
        assert (
            subproblem.subproblem_state.constraints.split_state is not None
        ), "Queue should only hold solutions with a valid split state."

        if self.global_prima_constraints is not None:
            subproblem.set_prima_coefficients(
                self.global_prima_constraints.prima_coefficients
            )
        self._insert_sorted_into(subproblem, self._not_yet_split_queue)

    def _insert_sorted_into(
        self,
        subproblem: ReadonlyVerificationSubproblem,
        subproblem_list: List[ReadonlyVerificationSubproblem],
    ) -> None:
        sorted_index = 0
        for current_subproblem in subproblem_list:
            if current_subproblem.lower_bound >= subproblem.lower_bound:
                break
            sorted_index += 1
        subproblem_list.insert(sorted_index, subproblem)
