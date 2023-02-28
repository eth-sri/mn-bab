from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor

from src.state.constraints import Constraints, ReadonlyConstraints
from src.state.layer_bounds import LayerBounds, ReadonlyLayerBounds
from src.state.parameters import (
    Parameters,
    ParametersForQuery,
    ReadonlyParameters,
    ReadonlyParametersForQuery,
)
from src.state.prima_constraints import PrimaConstraints, ReadonlyPrimaConstraints
from src.state.split_state import ReadonlySplitState, SplitState
from src.state.subproblem_state import ReadonlySubproblemState, SubproblemState
from src.state.tags import LayerTag, ParameterTag, QueryTag
from src.verification_subproblem import (
    ReadonlyVerificationSubproblem,
    VerificationSubproblem,
)


def batch_subproblems(
    subproblems: Sequence[ReadonlyVerificationSubproblem],
    reuse_single_subproblem: bool,  # (typically True, makes the behavior explicit at the call site)
) -> VerificationSubproblem:
    if len(subproblems) == 1 and reuse_single_subproblem:
        assert isinstance(subproblems[0], VerificationSubproblem)
        return subproblems[0]  # reuse parameters from parent subproblem
    assert not any(
        subproblem.is_infeasible for subproblem in subproblems
    ), "Infeasible subproblem in subproblems to be batched."
    device = subproblems[0].device

    assert all(
        subproblem.device == device for subproblem in subproblems
    ), "All subproblems must be on the same device in order to be batched."

    lower_bound = min([subproblem.lower_bound for subproblem in subproblems])
    upper_bound = min([subproblem.upper_bound for subproblem in subproblems])

    subproblem_states = [
        subproblem.subproblem_state for subproblem in subproblems
    ]  # TODO: use generators instead of lists?

    batched_subproblem_state = _batch_subproblem_states(subproblem_states, device)

    return VerificationSubproblem(
        lower_bound,
        upper_bound,
        batched_subproblem_state,
        device,
    )


def unbatch_subproblems(
    subproblem_state: SubproblemState,
    improved_lbs: Sequence[
        float
    ],  # TODO: put Sequence[float]s into VerificationSubproblem directly (just like for number_of_nodes_split)
    improved_ubs: Sequence[float],
    reset_intermediate_layer_bounds_to_be_kept_fixed: bool,  # TODO: why is this even done?
) -> Sequence[VerificationSubproblem]:

    batch_size = subproblem_state.batch_size
    assert len(improved_lbs) == batch_size
    assert len(improved_ubs) == batch_size

    device = subproblem_state.device

    subproblem_states = _unbatch_subproblem_states(
        subproblem_state,
        reset_intermediate_layer_bounds_to_be_kept_fixed,
    )

    return [
        VerificationSubproblem(
            improved_lbs[i],
            improved_ubs[i],
            subproblem_states[i],
            device,
        )
        for i in range(batch_size)
    ]


def _batch_subproblem_states(
    subproblem_states: Sequence[ReadonlySubproblemState],
    device: torch.device,
) -> SubproblemState:
    batch_size = sum(
        subproblem_state.batch_size for subproblem_state in subproblem_states
    )
    constraints_s = [
        subproblem_state.constraints for subproblem_state in subproblem_states
    ]
    parameters_s = [
        subproblem_state.parameters for subproblem_state in subproblem_states
    ]

    constraints = _batch_constraints(constraints_s, device)
    parameters = _batch_parameters(parameters_s, device)

    return SubproblemState(constraints, parameters, batch_size, device)


def _unbatch_subproblem_states(
    subproblem_state: ReadonlySubproblemState,
    reset_intermediate_layer_bounds_to_be_kept_fixed: bool,
) -> Sequence[SubproblemState]:
    batch_size = subproblem_state.batch_size

    constraints_s = _unbatch_constraints(
        subproblem_state.constraints,
        reset_intermediate_layer_bounds_to_be_kept_fixed,
    )
    parameters_s = _unbatch_parameters(subproblem_state.parameters)

    device = subproblem_state.device

    return [
        SubproblemState(
            constraints=constraints_s[i],
            parameters=parameters_s[i],
            batch_size=1,
            device=device,
        )
        for i in range(batch_size)
    ]


def _batch_constraints(
    constraints_s: Sequence[ReadonlyConstraints],
    device: torch.device,
) -> Constraints:
    batch_size = sum(constraints.batch_size for constraints in constraints_s)
    if all(constraints.split_state is not None for constraints in constraints_s):
        split_states: Optional[Sequence[ReadonlySplitState]] = [constraints.split_state for constraints in constraints_s]  # type: ignore[misc] # mypy can't see split_state is not None
    else:
        assert all(
            constraints.split_state is None for constraints in constraints_s
        ), "incompatible split states for batching"
        split_states = None
    layer_bounds_s = [constraints.layer_bounds for constraints in constraints_s]

    if all(constraints.prima_constraints is not None for constraints in constraints_s):

        prima_constraints_s: Optional[Sequence[ReadonlyPrimaConstraints]] = [constraints.prima_constraints for constraints in constraints_s]  # type: ignore[misc] # mypy can't see prima_constraints is not None
    else:
        assert all(
            constraints.prima_constraints is None for constraints in constraints_s
        ), "incompatible prima constraints for batching"
        prima_constraints_s = None

    split_state = (
        _batch_split_states(split_states, device) if split_states is not None else None
    )
    layer_bounds = _batch_layer_bounds(layer_bounds_s, device)
    prima_constraints = (
        _batch_prima_constraints(prima_constraints_s, device)
        if prima_constraints_s is not None
        else None
    )

    is_infeasible = torch.cat(
        tuple(constraints.is_infeasible for constraints in constraints_s)
    )

    return Constraints(
        split_state=split_state,
        layer_bounds=layer_bounds,
        prima_constraints=prima_constraints,
        is_infeasible=is_infeasible,  # TODO: copy may be unnecessary
        batch_size=batch_size,
        device=device,
    )


def _unbatch_constraints(
    constraints: ReadonlyConstraints,
    reset_intermediate_layer_bounds_to_be_kept_fixed: bool,
) -> Sequence[Constraints]:

    batch_size = constraints.batch_size

    split_states = (
        _unbatch_split_states(constraints.split_state)
        if constraints.split_state is not None
        else None
    )
    layer_bounds_s = _unbatch_layer_bounds(
        constraints.layer_bounds,
        reset_intermediate_layer_bounds_to_be_kept_fixed,
    )
    prima_constraints_s = (
        _unbatch_prima_constraints(constraints.prima_constraints)
        if constraints.prima_constraints is not None
        else None
    )

    is_infeasibles = constraints.is_infeasible

    device = constraints.device

    return [
        Constraints(
            split_state=split_states[i] if split_states is not None else None,
            layer_bounds=layer_bounds_s[i],
            prima_constraints=prima_constraints_s[i]
            if prima_constraints_s is not None
            else None,
            is_infeasible=is_infeasibles[i].unsqueeze(0),  # TODO: copy unnecessary
            batch_size=1,
            device=device,
        )
        for i in range(batch_size)
    ]


def _batch_split_states(
    split_states: Sequence[ReadonlySplitState],
    device: torch.device,
) -> SplitState:
    batch_size = sum(split_state.batch_size for split_state in split_states)

    split_constraints = _compress_to_batch(
        [split_state.split_constraints for split_state in split_states]
    )
    split_points = _compress_to_batch(
        [split_state.split_points for split_state in split_states]
    )
    number_of_nodes_split = [
        n for split_state in split_states for n in split_state.number_of_nodes_split
    ]

    return SplitState(
        split_constraints, split_points, number_of_nodes_split, batch_size, device
    )


def _unbatch_split_states(
    split_state: ReadonlySplitState,
) -> Sequence[SplitState]:
    batch_size = split_state.batch_size

    split_constraints_s = _unbatch_layer_property(
        split_state.split_constraints, batch_size
    )
    split_points_s = _unbatch_layer_property(split_state.split_points, batch_size)

    device = split_state.device

    return [
        SplitState(
            split_constraints=split_constraints_s[i],
            split_points=split_points_s[i],
            number_of_nodes_split=[split_state.number_of_nodes_split[i]],
            batch_size=1,
            device=device,
        )
        for i in range(batch_size)
    ]


def _batch_layer_bounds(
    layer_bounds_s: Sequence[ReadonlyLayerBounds],
    device: torch.device,
) -> LayerBounds:
    batch_size = sum(layer_bounds.batch_size for layer_bounds in layer_bounds_s)

    intermediate_layer_bounds_to_be_kept_fixed = layer_bounds_s[
        0
    ].intermediate_layer_bounds_to_be_kept_fixed
    assert all(
        layer_bounds.intermediate_layer_bounds_to_be_kept_fixed
        == intermediate_layer_bounds_to_be_kept_fixed
        for layer_bounds in layer_bounds_s
    ), "All subproblems must agree on which intermediate layers are kept fixed."

    intermediate_bounds = _compress_intermediate_bounds_to_batch(
        [layer_bounds.intermediate_bounds for layer_bounds in layer_bounds_s]
    )

    return LayerBounds(
        intermediate_layer_bounds_to_be_kept_fixed=intermediate_layer_bounds_to_be_kept_fixed,
        intermediate_bounds=intermediate_bounds,
        batch_size=batch_size,
        device=device,
    )


def _unbatch_layer_bounds(
    layer_bounds: ReadonlyLayerBounds,
    reset_intermediate_layer_bounds_to_be_kept_fixed: bool,
) -> Sequence[LayerBounds]:
    batch_size = layer_bounds.batch_size

    intermediate_bounds_s = _unbatch_intermediate_bounds(
        layer_bounds.intermediate_bounds, batch_size
    )

    device = layer_bounds.device

    return [
        LayerBounds(
            intermediate_layer_bounds_to_be_kept_fixed=[]
            if reset_intermediate_layer_bounds_to_be_kept_fixed
            else layer_bounds.intermediate_layer_bounds_to_be_kept_fixed,
            intermediate_bounds=intermediate_bounds_s[i],
            batch_size=1,
            device=device,
        )
        for i in range(batch_size)
    ]


def _batch_prima_constraints(
    prima_constraints_s: Sequence[ReadonlyPrimaConstraints],
    device: torch.device,
) -> PrimaConstraints:
    batch_size = sum(
        prima_constraints.batch_size for prima_constraints in prima_constraints_s
    )

    prima_coefficients = _batch_prima_coefficient_dicts(
        [
            prima_constraints.prima_coefficients
            for prima_constraints in prima_constraints_s
        ],
        device,
    )
    return PrimaConstraints(prima_coefficients, batch_size, device)


def _unbatch_prima_constraints(
    prima_constraints: ReadonlyPrimaConstraints,
) -> Sequence[PrimaConstraints]:
    batch_size = prima_constraints.batch_size

    prima_coefficients_s = _unbatch_prima_coefficient_dicts(
        prima_constraints.prima_coefficients, batch_size
    )

    device = prima_constraints.device

    return [
        PrimaConstraints(
            prima_coefficients=prima_coefficients_s[i], batch_size=1, device=device
        )
        for i in range(batch_size)
    ]


def _batch_prima_coefficient_dicts(
    prima_coefficients_set: Sequence[Mapping[LayerTag, Tuple[Tensor, Tensor, Tensor]]],
    device: torch.device,
) -> Dict[LayerTag, Tuple[Tensor, Tensor, Tensor]]:
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


def _batch_parameters(
    parameters_s: Sequence[ReadonlyParameters],
    device: torch.device,
) -> Parameters:
    batch_size = sum(parameters.batch_size for parameters in parameters_s)

    parameters_by_query = _batch_parameter_dicts(
        [parameters.parameters_by_query for parameters in parameters_s],
        batch_size,
        device,
    )

    return Parameters(parameters_by_query, batch_size, device, use_params=True)


def _unbatch_parameters(
    parameters: ReadonlyParameters,
) -> Sequence[Parameters]:
    batch_size = parameters.batch_size
    device = parameters.device

    parameters_by_query_s = _unbatch_parameter_dicts(
        parameters.parameters_by_query, batch_size, device
    )

    return [
        Parameters(
            parameters_by_query=parameters_by_query_s[i],
            batch_size=1,
            device=device,
            use_params=True,
        )
        for i in range(batch_size)
    ]


def _batch_parameter_dicts(  # TODO: move some of this logic into src.state.parameters
    parameters_by_query: Sequence[Mapping[QueryTag, ReadonlyParametersForQuery]],
    batch_size: int,
    device: torch.device,
) -> Dict[QueryTag, ParametersForQuery]:
    all_query_ids: Iterable[QueryTag] = set().union(*parameters_by_query)
    parameters_by_query_batch: Dict[QueryTag, ParametersForQuery] = {}
    for query_id in all_query_ids:
        parameters_by_query_batch[query_id] = ParametersForQuery.create_default(
            batch_size=batch_size, device=device
        )
        all_param_keys: Iterable[ParameterTag] = set().union(
            *(
                params[query_id].parameters
                for params in parameters_by_query
                if query_id in params
            )
        )
        for param_key in all_param_keys:
            if "prima" in param_key:
                parameters_by_query_batch[query_id].parameters[param_key] = {}
                all_layer_ids: Iterable[LayerTag] = set().union(
                    *(
                        params[query_id].parameters[param_key]
                        for params in parameters_by_query
                        if query_id in params
                        and param_key in params[query_id].parameters
                    )
                )
                for layer_id in all_layer_ids:
                    max_number_of_constraints = max(
                        [
                            params[query_id].parameters[param_key][layer_id].shape[1]
                            for params in parameters_by_query
                        ]
                    )
                    parameters_by_query_batch[query_id].parameters[param_key][
                        layer_id
                    ] = torch.cat(
                        [
                            _pad_to_match_size_in_dim_one(
                                params[query_id].parameters[param_key][layer_id],
                                max_number_of_constraints,
                                device,
                            )
                            for params in parameters_by_query
                        ],
                        dim=0,
                    )
            else:
                parameters_by_query_batch[query_id].parameters[
                    param_key
                ] = _compress_to_batch(
                    [
                        params[query_id].parameters[param_key]
                        for params in parameters_by_query
                    ]
                )
    return parameters_by_query_batch


def _pad_to_match_size_in_dim_one(a: Tensor, size: int, device: torch.device) -> Tensor:
    current_size = a.shape[1]
    if current_size == size:
        return a
    assert size > current_size
    padding = torch.zeros(a.shape[0], size - current_size, *a.shape[2:], device=device)
    return torch.cat([a, padding], dim=1)


def _unbatch_parameter_dicts(
    parameters_by_query_batch: Mapping[QueryTag, ReadonlyParametersForQuery],
    batch_size: int,
    device: torch.device,
) -> Sequence[Dict[QueryTag, ParametersForQuery]]:
    unbatched_parameters: Sequence[Dict[QueryTag, ParametersForQuery]] = [
        {} for __ in range(batch_size)
    ]
    for query_id in parameters_by_query_batch:
        for parameters in unbatched_parameters:
            parameters[query_id] = ParametersForQuery.create_default(
                batch_size=batch_size, device=device
            )
        for param_key in parameters_by_query_batch[query_id].parameters:
            for parameters in unbatched_parameters:
                parameters[query_id].parameters[param_key] = {}
            for layer_id in parameters_by_query_batch[query_id].parameters[param_key]:
                split_layer_parameters = (
                    parameters_by_query_batch[query_id]
                    .parameters[param_key][layer_id]
                    .split(1, dim=0)
                )
                for i, parameters in enumerate(unbatched_parameters):
                    parameters[query_id].parameters[param_key][layer_id] = (
                        split_layer_parameters[i].clone().detach()
                    )
    return unbatched_parameters


def _unbatch_prima_coefficient_dicts(
    prima_coefficients_batch: Mapping[LayerTag, Tuple[Tensor, Tensor, Tensor]],
    batch_size: int,
) -> Sequence[Dict[LayerTag, Tuple[Tensor, Tensor, Tensor]]]:
    prima_coefficients_set: Sequence[Dict[LayerTag, Tuple[Tensor, Tensor, Tensor]]] = [
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


def _unbatch_layer_property(
    property_batch: Mapping[LayerTag, Tensor], batch_size: int
) -> Sequence[Dict[LayerTag, Tensor]]:
    properties: Sequence[Dict[LayerTag, Tensor]] = [{} for __ in range(batch_size)]
    for layer_id in property_batch:
        layer_property = property_batch[layer_id].split(1, dim=0)
        for i in range(batch_size):
            properties[i][layer_id] = layer_property[i]
    return properties


def _compress_to_batch(
    dicts: Sequence[Mapping[LayerTag, Tensor]]
) -> Dict[LayerTag, Tensor]:
    return {
        key: torch.cat(
            [d[key] for d in dicts],
            dim=0,
        )
        for d in dicts
        for key in d.keys()
    }


def _compress_intermediate_bounds_to_batch(
    intermediate_bounds: Sequence[Mapping[LayerTag, Tuple[Tensor, Tensor]]]
) -> OrderedDict[LayerTag, Tuple[Tensor, Tensor]]:
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


def _unbatch_intermediate_bounds(
    layer_bounds_batch: Mapping[LayerTag, Tuple[Tensor, Tensor]], batch_size: int
) -> Sequence[OrderedDict[LayerTag, Tuple[Tensor, Tensor]]]:
    layer_bounds: List[OrderedDict[LayerTag, Tuple[Tensor, Tensor]]] = [
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
