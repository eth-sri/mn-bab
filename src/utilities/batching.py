from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch import Tensor

from src.verification_subproblem import VerificationSubproblem


def batch_layer_properties(
    subproblems: Sequence[VerificationSubproblem],
) -> VerificationSubproblem:
    if len(subproblems) == 1:
        return subproblems[0]
    assert not any(
        [subproblem.is_infeasible for subproblem in subproblems]
    ), "Infeasible subproblem in subproblems to be batched."
    device = subproblems[0].device
    assert all(
        [subproblem.device == device for subproblem in subproblems]
    ), "All subproblems must be on the same device in order to be batched."
    intermediate_layers_to_be_kept_fixed = subproblems[
        0
    ].intermediate_layer_bounds_to_be_kept_fixed
    assert all(
        [
            subproblem.intermediate_layer_bounds_to_be_kept_fixed
            == intermediate_layers_to_be_kept_fixed
            for subproblem in subproblems
        ]
    ), "All subproblems must agree on which intermediate layers are kept fixed."

    lower_bound = min([subproblem.lower_bound for subproblem in subproblems])
    upper_bound = min([subproblem.upper_bound for subproblem in subproblems])
    split_constraints = _compress_to_batch(
        [subproblem.split_constraints for subproblem in subproblems]
    )
    intermediate_bounds = _compress_intermediate_bounds_to_batch(
        [subproblem.intermediate_bounds for subproblem in subproblems]
    )
    parameters_by_starting_layer = _batch_parameters(
        [subproblem.parameters_by_starting_layer for subproblem in subproblems], device
    )
    prima_coefficients = _batch_prima_coefficients(
        [subproblem.prima_coefficients for subproblem in subproblems], device
    )
    number_of_nodes_split = [
        n for subproblem in subproblems for n in subproblem.number_of_nodes_split
    ]

    return VerificationSubproblem(
        lower_bound,
        upper_bound,
        split_constraints,
        intermediate_layers_to_be_kept_fixed,
        intermediate_bounds,
        parameters_by_starting_layer,
        prima_coefficients,
        False,
        number_of_nodes_split,
        device,
    )


def _batch_parameters(
    parameters_by_starting_layer: Sequence[Dict[int, Dict[str, Dict[int, Tensor]]]],
    device: torch.device,
) -> Dict[int, Dict[str, Dict[int, Tensor]]]:
    all_starting_layer_ids: Iterable[int] = set().union(*parameters_by_starting_layer)  # type: ignore
    parameters_by_starting_layer_batch: Dict[int, Dict[str, Dict[int, Tensor]]] = {}
    for starting_layer_id in all_starting_layer_ids:
        parameters_by_starting_layer_batch[starting_layer_id] = {}
        all_param_keys: Iterable[str] = set().union(
            *[
                params[starting_layer_id]  # type: ignore
                for params in parameters_by_starting_layer
                if starting_layer_id in params
            ]
        )
        for param_key in all_param_keys:
            if "prima" in param_key:
                parameters_by_starting_layer_batch[starting_layer_id][param_key] = {}
                all_layer_ids: Iterable[int] = set().union(
                    *[
                        params[starting_layer_id][param_key]  # type: ignore
                        for params in parameters_by_starting_layer
                        if starting_layer_id in params
                        and param_key in params[starting_layer_id]
                    ]
                )
                for layer_id in all_layer_ids:
                    max_number_of_constraints = max(
                        [
                            params[starting_layer_id][param_key][layer_id].shape[1]
                            for params in parameters_by_starting_layer
                        ]
                    )
                    parameters_by_starting_layer_batch[starting_layer_id][param_key][
                        layer_id
                    ] = torch.cat(
                        [
                            _pad_to_match_size_in_dim_one(
                                params[starting_layer_id][param_key][layer_id],
                                max_number_of_constraints,
                                device,
                            )
                            for params in parameters_by_starting_layer
                        ],
                        dim=0,
                    )
            else:
                parameters_by_starting_layer_batch[starting_layer_id][
                    param_key
                ] = _compress_to_batch(
                    [
                        params[starting_layer_id][param_key]
                        for params in parameters_by_starting_layer
                    ]
                )
    return parameters_by_starting_layer_batch


def _pad_to_match_size_in_dim_one(a: Tensor, size: int, device: torch.device) -> Tensor:
    current_size = a.shape[1]
    if current_size == size:
        return a
    assert size > current_size
    padding = torch.zeros(a.shape[0], size - current_size, *a.shape[2:], device=device)
    return torch.cat([a, padding], dim=1)


def _pad_sparse_matrix_to_match_size_in_dim_two(
    a: Tensor, size: int, device: torch.device
) -> Tensor:
    current_size = a.shape[2]
    if current_size == size:
        return a
    assert size > current_size
    padding = torch.zeros(
        *a.shape[:2], size - current_size, *a.shape[3:], layout=a.layout, device=device
    )
    return torch.cat([a, padding], dim=2)


def _batch_prima_coefficients(
    prima_coefficients_set: Sequence[Dict[int, Tuple[Tensor, Tensor, Tensor]]],
    device: torch.device,
) -> Dict[int, Tuple[Tensor, Tensor, Tensor]]:
    reference_prima_coefficients = prima_coefficients_set[0]
    max_number_of_prima_constraints = {
        layer_id: max(
            [
                prima_coefs[layer_id][0].shape[2]
                for prima_coefs in prima_coefficients_set
            ]
        )
        for layer_id in reference_prima_coefficients
    }
    prima_coefficients_batch = {}
    for layer_id in reference_prima_coefficients:
        assert all(
            [
                layer_id in prima_coefficients
                for prima_coefficients in prima_coefficients_set
            ]
        )
        prima_output_coefs = torch.cat(
            [
                _pad_sparse_matrix_to_match_size_in_dim_two(
                    prima_coefficients[layer_id][0],
                    max_number_of_prima_constraints[layer_id],
                    device,
                )
                for prima_coefficients in prima_coefficients_set
            ],
            dim=0,
        )
        prima_input_coefs = torch.cat(
            [
                _pad_sparse_matrix_to_match_size_in_dim_two(
                    prima_coefficients[layer_id][1],
                    max_number_of_prima_constraints[layer_id],
                    device,
                )
                for prima_coefficients in prima_coefficients_set
            ],
            dim=0,
        )
        prima_const_coefs = torch.cat(
            [
                _pad_sparse_matrix_to_match_size_in_dim_two(
                    prima_coefficients[layer_id][2],
                    max_number_of_prima_constraints[layer_id],
                    device,
                )
                for prima_coefficients in prima_coefficients_set
            ],
            dim=0,
        )
        prima_coefficients_batch[layer_id] = (
            prima_output_coefs.coalesce()
            if prima_output_coefs.is_sparse
            else prima_output_coefs,
            prima_input_coefs.coalesce()
            if prima_input_coefs.is_sparse
            else prima_input_coefs,
            prima_const_coefs,
        )
    return prima_coefficients_batch


def unbatch_parameters(
    parameters_by_starting_layer_batch: Dict[int, Dict[str, Dict[int, Tensor]]],
    batch_size: int,
) -> Sequence[Dict[int, Dict[str, Dict[int, Tensor]]]]:
    unbatched_parameters: Sequence[Dict[int, Dict[str, Dict[int, Tensor]]]] = [
        {} for __ in range(batch_size)
    ]
    for starting_layer_id in parameters_by_starting_layer_batch:
        for parameters in unbatched_parameters:
            parameters[starting_layer_id] = {}
        for param_key in parameters_by_starting_layer_batch[starting_layer_id]:
            for parameters in unbatched_parameters:
                parameters[starting_layer_id][param_key] = {}
            for layer_id in parameters_by_starting_layer_batch[starting_layer_id][
                param_key
            ]:
                split_layer_parameters = parameters_by_starting_layer_batch[
                    starting_layer_id
                ][param_key][layer_id].split(1, dim=0)
                for i, parameters in enumerate(unbatched_parameters):
                    parameters[starting_layer_id][param_key][layer_id] = (
                        split_layer_parameters[i].clone().detach()
                    )
    return unbatched_parameters


def unbatch_prima_coefficients(
    prima_coefficients_batch: Dict[int, Tuple[Tensor, Tensor, Tensor]], batch_size: int
) -> Sequence[Dict[int, Tuple[Tensor, Tensor, Tensor]]]:
    prima_coefficients_set: Sequence[Dict[int, Tuple[Tensor, Tensor, Tensor]]] = [
        {} for __ in range(batch_size)
    ]
    for layer_id in prima_coefficients_batch:
        for i, prima_coefficients in enumerate(prima_coefficients_set):
            prima_output_coefficients_of_layer = prima_coefficients_batch[layer_id][0][
                i
            ].unsqueeze(0)
            prima_input_coefficients_of_layer = prima_coefficients_batch[layer_id][1][
                i
            ].unsqueeze(0)
            prima_const_coefficients_of_layer = prima_coefficients_batch[layer_id][2][
                i
            ].unsqueeze(0)
            prima_coefficients[layer_id] = (
                prima_output_coefficients_of_layer,
                prima_input_coefficients_of_layer,
                prima_const_coefficients_of_layer,
            )

    return prima_coefficients_set


def unbatch_layer_property(
    property_batch: Dict[int, Tensor], batch_size: int
) -> Sequence[Dict[int, Tensor]]:
    properties: Sequence[Dict[int, Tensor]] = [{} for __ in range(batch_size)]
    for layer_id in property_batch:
        layer_property = property_batch[layer_id].split(1, dim=0)
        for i in range(batch_size):
            properties[i][layer_id] = layer_property[i]
    return properties


def unbatch_layer_bounds(
    layer_bounds_batch: OrderedDict[int, Tuple[Tensor, Tensor]], batch_size: int
) -> Sequence[OrderedDict[int, Tuple[Tensor, Tensor]]]:
    layer_bounds: List[OrderedDict[int, Tuple[Tensor, Tensor]]] = [
        OrderedDict() for __ in range(batch_size)
    ]
    for layer_id in layer_bounds_batch:
        current_layer_lower_bounds = layer_bounds_batch[layer_id][0].split(1, dim=0)
        current_layer_upper_bounds = layer_bounds_batch[layer_id][1].split(1, dim=0)
        for i in range(batch_size):
            layer_bounds[i][layer_id] = (
                current_layer_lower_bounds[i],
                current_layer_upper_bounds[i],
            )
    return layer_bounds


def _compress_to_batch(dicts: Sequence[Dict[int, Tensor]]) -> Dict[int, Tensor]:
    return {
        key: torch.cat(
            [d[key] for d in dicts],
            dim=0,
        )
        for d in dicts
        for key in d.keys()
    }


def _compress_intermediate_bounds_to_batch(
    intermediate_bounds: Sequence[OrderedDict[int, Tuple[Tensor, Tensor]]]
) -> OrderedDict[int, Tuple[Tensor, Tensor]]:
    batched_intermediate_bounds = OrderedDict()
    reference_keys = intermediate_bounds[0].keys()
    for key in reference_keys:
        batched_intermediate_bounds[key] = (
            torch.cat(
                [ib[key][0] for ib in intermediate_bounds],
                dim=0,
            ),
            torch.cat(
                [ib[key][1] for ib in intermediate_bounds],
                dim=0,
            ),
        )
    return batched_intermediate_bounds
