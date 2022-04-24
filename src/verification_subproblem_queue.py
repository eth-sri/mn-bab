from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from src.verification_subproblem import VerificationSubproblem


class VerificationSubproblemQueue:
    """Priority queue of VerificationSubproblems sorted by lower bound in ascending order."""

    def __init__(
        self, initial_subproblem: VerificationSubproblem, batch_sizes: Sequence[int]
    ) -> None:
        self._layer_id_to_index = {}
        for index, layer_id in enumerate(initial_subproblem.split_constraints):
            self._layer_id_to_index[layer_id] = index
        self._batch_sizes = batch_sizes
        self._not_yet_split_queue: List[VerificationSubproblem] = []
        self._queues_by_intermediate_bounds_to_keep: Sequence[
            List[VerificationSubproblem]
        ] = [[] for __ in range(len(batch_sizes))]
        self.global_prima_coefficients = initial_subproblem.prima_coefficients
        self.insert_sorted(initial_subproblem)

    def __len__(self) -> int:
        return len(self._not_yet_split_queue) + int(
            sum(
                [
                    (len(queue) / 2)
                    for queue in self._queues_by_intermediate_bounds_to_keep
                ]
            )
        )

    def peek(self) -> Optional[VerificationSubproblem]:
        if not self._not_yet_split_queue and not any(
            self._queues_by_intermediate_bounds_to_keep
        ):
            return None
        minimum_lower_bound_not_yet_split = (
            self._not_yet_split_queue[0].lower_bound
            if self._not_yet_split_queue
            else float("inf")
        )
        minimum_lower_bounds_already_split = [
            subproblem_queue[0].lower_bound if subproblem_queue else float("inf")
            for subproblem_queue in self._queues_by_intermediate_bounds_to_keep
        ]
        if minimum_lower_bound_not_yet_split < min(minimum_lower_bounds_already_split):
            return self._not_yet_split_queue[0]
        else:
            index_of_minimum = np.argmin(minimum_lower_bounds_already_split)
            return self._queues_by_intermediate_bounds_to_keep[index_of_minimum][0]

    def pop(
        self,
        find_node_to_split: Callable[[VerificationSubproblem], Tuple[int, ...]],
        recompute_intermediate_bounds_after_branching: bool,
    ) -> Sequence[VerificationSubproblem]:
        minimum_lower_bound_not_yet_split = (
            self._not_yet_split_queue[0].lower_bound
            if self._not_yet_split_queue
            else float("inf")
        )
        minimum_lower_bounds_already_split = [
            subproblem_queue[0].lower_bound if subproblem_queue else float("inf")
            for subproblem_queue in self._queues_by_intermediate_bounds_to_keep
        ]
        if min(minimum_lower_bounds_already_split) < minimum_lower_bound_not_yet_split:
            index_of_last_intermediate_bounds_kept = np.argmin(
                minimum_lower_bounds_already_split
            )
        else:
            minimum_subproblem = self._not_yet_split_queue[0]
            if not minimum_subproblem.is_fully_split:
                del self._not_yet_split_queue[0]
                node_to_split = find_node_to_split(minimum_subproblem)
                index_of_last_intermediate_bounds_kept = (
                    self._layer_id_to_index[node_to_split[0]]
                    if recompute_intermediate_bounds_after_branching
                    else -1
                )
                (
                    negatively_split_subproblem,
                    positively_split_subproblem,
                ) = minimum_subproblem.split(
                    node_to_split, recompute_intermediate_bounds_after_branching
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
            current_split_index_of_last_intermediate_bounds_kept = (
                self._layer_id_to_index[node_to_split[0]]
                if recompute_intermediate_bounds_after_branching
                else -1
            )
            (
                negatively_split_subproblem,
                positively_split_subproblem,
            ) = subproblem_to_split.split(
                node_to_split, recompute_intermediate_bounds_after_branching
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
    ) -> Sequence[VerificationSubproblem]:
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
        subproblem: VerificationSubproblem,
    ) -> None:
        assert (
            not subproblem.is_infeasible
        ), "Queue should only hold feasible subproblems."
        # TODO: possibly assert that prima coefficients are the same
        subproblem.prima_coefficients = self.global_prima_coefficients
        self._insert_sorted_into(subproblem, self._not_yet_split_queue)

    def _insert_sorted_into(
        self,
        subproblem: VerificationSubproblem,
        subproblem_list: List[VerificationSubproblem],
    ) -> None:
        sorted_index = 0
        for current_subproblem in subproblem_list:
            if current_subproblem.lower_bound >= subproblem.lower_bound:
                break
            sorted_index += 1
        subproblem_list.insert(sorted_index, subproblem)
