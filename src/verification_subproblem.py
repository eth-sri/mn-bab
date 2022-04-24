from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from typing import Any, Dict, Sequence, Tuple

import torch
from torch import Tensor


class VerificationSubproblem:
    def __init__(
        self,
        lower_bound: float,
        upper_bound: float,
        split_constraints: Dict[int, Tensor],
        intermediate_layer_bounds_to_be_kept_fixed: Sequence[int],
        intermediate_bounds: OrderedDict[int, Tuple[Tensor, Tensor]],
        parameters_by_starting_layer: Dict[int, Dict[str, Dict[int, Tensor]]],
        prima_coefficients: Dict[int, Tuple[Tensor, Tensor, Tensor]],
        is_infeasible: bool,
        number_of_nodes_split: Sequence[int],
        device: torch.device = torch.device("cpu"),
    ):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.split_constraints = split_constraints
        self.intermediate_layer_bounds_to_be_kept_fixed = (
            intermediate_layer_bounds_to_be_kept_fixed
        )
        self.intermediate_bounds = intermediate_bounds
        self.parameters_by_starting_layer = parameters_by_starting_layer
        self.prima_coefficients = prima_coefficients
        self.is_infeasible = is_infeasible
        self.number_of_nodes_split = number_of_nodes_split
        self.is_fully_split = all(
            [(splits != 0).all() for splits in split_constraints.values()]
        )
        self.device = device

        self.to(self.device)

    @classmethod
    def create_default(
        cls,
        lower_bound: float = -float("inf"),
        upper_bound: float = float("inf"),
        split_constraints: Dict[int, Tensor] = {},
    ) -> VerificationSubproblem:
        return cls(
            lower_bound,
            upper_bound,
            split_constraints,
            [],
            OrderedDict(),
            {},
            {},
            False,
            [0],
        )

    def to(self, device: torch.device) -> None:
        self.device = device
        self.split_constraints = self._move_to(self.split_constraints, device)
        self.intermediate_bounds = self._move_to(self.intermediate_bounds, device)
        self.parameters_by_starting_layer = self._move_to(
            self.parameters_by_starting_layer, device
        )
        self.prima_coefficients = self._move_to(self.prima_coefficients, device)

    def _move_to(self, obj: Any, device: torch.device) -> Any:
        if torch.is_tensor(obj):
            return obj.to(device)
        elif isinstance(obj, dict):
            res: Any = {}
            for k, v in obj.items():
                res[k] = self._move_to(v, device)
            return res
        elif isinstance(obj, list):
            res = []
            for v in obj:
                res.append(self._move_to(v, device))
            return res
        elif isinstance(obj, tuple):
            res = []
            for v in obj:
                res.append(self._move_to(v, device))
            return tuple(res)
        else:
            raise TypeError("Invalid type for move_to")

    def split(
        self,
        node_to_split: Tuple[int, ...],
        recompute_intermediate_bounds_after_branching: bool,
    ) -> Tuple[VerificationSubproblem, VerificationSubproblem]:
        negative_split, positive_split = self._compute_new_split_constraints(
            node_to_split
        )

        (
            intermediate_bounds_for_negative_split,
            intermediate_bounds_for_positive_split,
        ) = self._update_intermediate_bounds(node_to_split)

        intermediate_layer_bounds_to_be_kept_fixed = []
        if recompute_intermediate_bounds_after_branching:
            for layer_id in self.intermediate_bounds:
                intermediate_layer_bounds_to_be_kept_fixed.append(layer_id)
                if layer_id == node_to_split[0]:
                    break
        else:
            intermediate_layer_bounds_to_be_kept_fixed = list(
                self.intermediate_bounds.keys()
            )

        active_parameters = (
            {
                starting_layer_id: parameters
                for starting_layer_id, parameters in self.parameters_by_starting_layer.items()
                if starting_layer_id not in intermediate_layer_bounds_to_be_kept_fixed
            }
            if not recompute_intermediate_bounds_after_branching
            else self.parameters_by_starting_layer
        )

        number_of_nodes_split = [n + 1 for n in self.number_of_nodes_split]

        negative_split_subproblem = VerificationSubproblem(
            self.lower_bound,
            self.upper_bound,
            negative_split,
            intermediate_layer_bounds_to_be_kept_fixed,
            intermediate_bounds_for_negative_split,
            active_parameters,
            self.prima_coefficients,
            self.is_infeasible,
            number_of_nodes_split,
        )
        positive_split_subproblem = VerificationSubproblem(
            self.lower_bound,
            self.upper_bound,
            positive_split,
            intermediate_layer_bounds_to_be_kept_fixed,
            intermediate_bounds_for_positive_split,
            active_parameters,
            self.prima_coefficients,
            self.is_infeasible,
            number_of_nodes_split,
        )

        return negative_split_subproblem, positive_split_subproblem

    def _compute_new_split_constraints(
        self, node_to_split: Tuple[int, ...]
    ) -> Tuple[Dict[int, Tensor], Dict[int, Tensor]]:
        layer_id = node_to_split[0]
        index_in_layer = 0, *node_to_split[1:]
        assert (
            self.split_constraints[layer_id][index_in_layer] == 0
        ), "Attempted to split a node that is already split."

        negative_split = deepcopy(self.split_constraints)
        positive_split = deepcopy(self.split_constraints)

        layer_split_constraints = self.split_constraints[layer_id]
        assert layer_split_constraints.shape[0] == 1

        neg_layer_split_constraints = layer_split_constraints.clone()
        neg_layer_split_constraints[index_in_layer] = 1
        negative_split[layer_id] = neg_layer_split_constraints

        pos_layer_split_constraints = layer_split_constraints.clone()
        pos_layer_split_constraints[index_in_layer] = -1
        positive_split[layer_id] = pos_layer_split_constraints

        return negative_split, positive_split

    def _update_intermediate_bounds(
        self,
        node_to_split: Tuple[int, ...],
    ) -> Tuple[
        OrderedDict[int, Tuple[Tensor, Tensor]],
        OrderedDict[int, Tuple[Tensor, Tensor]],
    ]:
        layer_id = node_to_split[0]
        index_in_layer = 0, *node_to_split[1:]
        assert (
            self.intermediate_bounds[layer_id][0][index_in_layer] < 0
            and self.intermediate_bounds[layer_id][1][index_in_layer] > 0
        ), "Attempted to split a stable node."

        intermediate_bounds_for_negative_split = deepcopy(self.intermediate_bounds)
        intermediate_bounds_for_positive_split = deepcopy(self.intermediate_bounds)

        layer_upper_bounds_for_negative_split = intermediate_bounds_for_negative_split[
            layer_id
        ][1].clone()
        layer_upper_bounds_for_negative_split[index_in_layer] = 0.0
        intermediate_bounds_for_negative_split[layer_id] = (
            intermediate_bounds_for_negative_split[layer_id][0],
            layer_upper_bounds_for_negative_split,
        )

        layer_lower_bounds_for_positive_split = intermediate_bounds_for_positive_split[
            layer_id
        ][0].clone()
        layer_lower_bounds_for_positive_split[index_in_layer] = 0.0
        intermediate_bounds_for_positive_split[layer_id] = (
            layer_lower_bounds_for_positive_split,
            intermediate_bounds_for_positive_split[layer_id][1],
        )

        return (
            intermediate_bounds_for_negative_split,
            intermediate_bounds_for_positive_split,
        )
