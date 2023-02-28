import itertools
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class OutputPropertyForm:

    """
    Represents an output property-formula in CNF.
    Each atom of the formula corresponds to a gt_tuple of the form (a,b,c) <=> a - b >= c
    property_matrix: a Tensor containing the query-coef for all sub-formulas row-wise
    property_threshold: a Tensor containing the c values for each gt_tuple
    combination_matrix: a num_clausels x num_gt_tuple matrix containing 1 in entry (i,j) if clausel i contains gt_tuple j
    disjunction_adapters: for every non-trivial disjunctive clause contains the linear+relu encoding to jointly optimize the network for this clause
    """

    def __init__(
        self,
        properties_to_verify: List[List[Tuple[int, int, float]]],
        property_matrix: Tensor,
        property_threshold: Tensor,
        combination_matrix: Tensor,
        disjunction_adapter: Optional[nn.Sequential],
        use_disj_adapter: bool,
        n_class: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.properties_to_verify = properties_to_verify
        self.property_matrix = property_matrix
        self.property_threshold = property_threshold
        self.combination_matrix = combination_matrix
        self.disjunction_adapter = disjunction_adapter
        self.use_disj_adapter = use_disj_adapter
        self.has_counter_example: bool = False
        self.n_class = n_class
        self.device = device
        self.dtype = dtype

    @classmethod
    def create_from_properties(
        cls,
        properties_to_verify: List[List[Tuple[int, int, float]]],
        disjunction_adapter: Optional[nn.Sequential],
        use_disj_adapter: bool,
        n_class: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "OutputPropertyForm":

        # properties_to_verify = sorted(properties_to_verify)

        all_gt_tuples = list(
            dict.fromkeys([*itertools.chain.from_iterable(properties_to_verify)])
        )

        gt_map = {x: i for i, x in enumerate(all_gt_tuples)}

        n_constraints = len(all_gt_tuples)
        property_matrix = torch.zeros(
            (n_constraints, n_class), device=device, dtype=dtype
        )
        property_threshold = torch.zeros((n_constraints,), device=device, dtype=dtype)
        combination_matrix = torch.zeros(
            (len(properties_to_verify), n_constraints), device=device, dtype=dtype
        )
        # These compute the disjunction a > x_1 || a - b > x_2 as ReLU(a-x_1) + ReLU(a-b-x_2) > 0

        # Check whether we have a true disjunction
        has_proper_disjunction = True in [
            len(clausel) > 1 for clausel in properties_to_verify
        ]
        if has_proper_disjunction:
            assert disjunction_adapter is None

        for property in all_gt_tuples:
            if property[0] != -1:
                property_matrix[gt_map[property], property[0]] = 1
                property_threshold[gt_map[property]] = torch.as_tensor(property[2])
            else:
                property_threshold[gt_map[property]] = -torch.as_tensor(property[2])
            if property[1] != -1:
                property_matrix[gt_map[property], property[1]] = -1

        for and_property_counter, and_property in enumerate(properties_to_verify):
            for or_property_counter, or_property in enumerate(and_property):
                combination_matrix[and_property_counter, gt_map[or_property]] = 1

        if use_disj_adapter and has_proper_disjunction:
            dis_layer = nn.Linear(n_class, n_constraints, bias=True)
            dis_layer.weight.data = property_matrix
            dis_layer.bias.data = -property_threshold + 1e-5
            n_class = len(properties_to_verify)

            sum_layer = nn.Linear(
                n_constraints, len(properties_to_verify), bias=True
            )  # One output per property
            sum_layer.weight.data = combination_matrix
            sum_layer.bias.data = torch.zeros_like(
                sum_layer.bias
            ) - 1e-5 * combination_matrix.sum(1).to(sum_layer.bias.device)

            disjunction_adapter = nn.Sequential(*[dis_layer, nn.ReLU(), sum_layer])
            disjunction_adapter.requires_grad_(False)

            property_matrix = torch.eye(
                len(properties_to_verify), device=device, dtype=dtype
            )
            property_threshold = torch.zeros(
                len(properties_to_verify), device=device, dtype=dtype
            )
            combination_matrix = torch.eye(
                len(properties_to_verify), device=device, dtype=dtype
            )

            properties_to_verify = [
                [(i, -1, 0)] for i, prop in enumerate(properties_to_verify)
            ]

        return cls(
            properties_to_verify,
            property_matrix.unsqueeze(0),
            property_threshold.unsqueeze(0),
            combination_matrix.unsqueeze(0),
            disjunction_adapter,
            use_disj_adapter,
            n_class,
            device,
            dtype,
        )

    def update_properties_to_verify(
        self,
        verified: Tensor,
        falsified: Tensor,
        query_lbs: Tensor,
        true_ub: bool,
        ignore_and_falsified: bool = False,
        easiest_and_first: bool = True,
    ) -> "OutputPropertyForm":
        new_properties_to_verify = []
        and_properties_verified = (
            torch.einsum(
                "bij,bj -> bi",
                self.combination_matrix,
                verified.to(self.combination_matrix.dtype),
            )
            >= 1
        )
        and_properties_falsified = torch.einsum(
            "bij,bj -> bi",
            self.combination_matrix,
            falsified.to(self.combination_matrix.dtype),
        ) == self.combination_matrix.sum(-1)
        if not true_ub:
            # Different or clauses might have been falsified for different points
            and_properties_falsified = torch.where(
                self.combination_matrix.sum(-1) == 1,
                and_properties_falsified,
                torch.zeros_like(and_properties_falsified),
            )
        if and_properties_falsified.any(1).all(0) and not ignore_and_falsified:
            # counterexample will be found by standard inference
            self.has_counter_example = True
            return self

        and_queries_lbs = []
        for and_property_counter, and_property in enumerate(self.properties_to_verify):
            if and_properties_verified[:, and_property_counter].all():
                continue
            if (
                self.combination_matrix[0, and_property_counter].unsqueeze(0)
                == self.combination_matrix[:, and_property_counter]
            ).all():
                and_property_filtered = []
                or_queries_lbs = []
                and_property_idx = self.combination_matrix[
                    0, and_property_counter
                ].bool()
                for or_property, falsified_prop, query_lb in zip(
                    and_property,
                    falsified.all(0)[and_property_idx],
                    query_lbs.min(0)[0][and_property_idx],
                ):
                    if falsified_prop and true_ub:
                        continue
                    and_property_filtered.append(or_property)
                    or_queries_lbs.append(query_lb.item())
            else:
                and_property_filtered = and_property
                or_queries_lbs = query_lbs.min(0)[0][
                    self.combination_matrix[0, and_property_counter].bool()
                ].tolist()
            and_property_filtered = [
                x
                for _, x in sorted(
                    zip(or_queries_lbs, and_property_filtered), reverse=True
                )
            ]  # "easiest" properties first => verificaiton
            and_queries_lbs.append(
                query_lbs[self.combination_matrix[:, and_property_counter].bool()].min()
            )
            new_properties_to_verify.append(and_property_filtered)
        new_properties_to_verify = [
            x
            for _, x in sorted(
                zip(and_queries_lbs, new_properties_to_verify),
                reverse=easiest_and_first,
            )
        ]  # "hardest" and properties first for better falsification
        # Update
        return OutputPropertyForm.create_from_properties(
            new_properties_to_verify,
            self.disjunction_adapter,
            self.use_disj_adapter,
            self.n_class,
            self.device,
            self.dtype,
        )

    def get_unit_clauses(self) -> List[Tuple[int, List[Tuple[int, int, float]]]]:
        unit_clauses = [
            (i, clause)
            for i, clause in enumerate(self.properties_to_verify)
            if len(clause) == 1
        ]
        return unit_clauses
