import functools
import itertools
from typing import Callable, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

from src.utilities.config import DomainSplittingConfig


def all_larger_equal(seq: Union[Sequence, Tensor], threshold: float) -> bool:
    return all(el >= threshold for el in seq)


def any_smaller(seq: Union[Sequence, Tensor], threshold: float) -> bool:
    return any(el < threshold for el in seq)


def get_neg_pos_comp(x: Tensor) -> Tuple[Tensor, Tensor]:
    neg_comp = torch.where(x < 0, x, torch.zeros_like(x))
    pos_comp = torch.where(x >= 0, x, torch.zeros_like(x))
    return neg_comp, pos_comp


def property_matrix_from_properties(
    properties_to_verify: List[List[Tuple[int, int, float]]],
    n_class: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Tensor, Tensor, Tensor]:

    all_gt_tuples = list(set([*itertools.chain.from_iterable(properties_to_verify)]))
    gt_map = {x: i for i, x in enumerate(all_gt_tuples)}

    n_constraints = len(all_gt_tuples)
    property_matrix = torch.zeros((n_constraints, n_class), device=device, dtype=dtype)
    property_threshold = torch.zeros((n_constraints,), device=device, dtype=dtype)
    combination_matrix = torch.zeros(
        (len(properties_to_verify), n_constraints), device=device, dtype=dtype
    )

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

    return (
        property_matrix.unsqueeze(0),
        property_threshold.unsqueeze(0),
        combination_matrix.unsqueeze(0),
    )


def update_propertiy_matrices(
    verified: Tensor,
    falsified: Tensor,
    property_matrix: Tensor,
    property_threshold: Tensor,
    combination_matrix: Tensor,
    true_ub: bool,
) -> Tuple[Tensor, Tensor]:
    and_properties_verified = (
        torch.einsum(
            "bij,bj -> bi", combination_matrix, verified.to(combination_matrix.dtype)
        )
        >= 1
    )
    and_properties_falsified = torch.einsum(
        "bij,bj -> bi", combination_matrix, falsified.to(combination_matrix.dtype)
    ) == combination_matrix.sum(-1)
    if not true_ub:
        # Different or clauses might have been falsified for different points
        and_properties_falsified = torch.where(
            combination_matrix.sum(-1) == 1,
            and_properties_falsified,
            torch.zeros_like(and_properties_falsified),
        )
    assert not and_properties_falsified.__and__(and_properties_verified).any()
    # constraints_verified = (
    #     (and_properties_verified.unsqueeze(2) * combination_matrix).sum(1).bool()
    # )

    property_matrix[verified] = 0
    property_threshold[verified] = -1

    if true_ub:
        property_matrix[falsified] = 0
        property_threshold[falsified] = 1

    return (
        and_properties_verified.all(1),
        and_properties_falsified.any(1),
    )


def compute_initial_splits(
    input_lb: Tensor,
    input_ub: Tensor,
    property_matrix: Tensor,
    property_threshold: Tensor,
    combination_matrix: Tensor,
    domain_splitting: DomainSplittingConfig,
) -> List[
    Tuple[
        Tensor,
        Tensor,
        Tuple[Tensor, Tensor, Tensor],
        int,
        Optional[Sequence[Sequence[Tuple[int, int, float]]]],
    ]
]:
    un_tight = (input_lb - input_ub).abs() > 1e-6
    n_splits = min(
        domain_splitting.initial_splits,
        int(domain_splitting.batch_size ** (1 / un_tight.sum()) + 0.5),
    )

    # assert input_lb.dim() == 2
    split_dims = domain_splitting.initial_split_dims.copy()
    if domain_splitting.initial_splits > 0:
        if len(split_dims) == 0:
            split_dims = un_tight.flatten().nonzero().flatten().tolist()
        initial_input_regions = split_input_regions(
            [(input_lb, input_ub)],
            dim=split_dims,
            splits=[n_splits] * (input_lb.shape[-1]),
        )
    else:
        initial_input_regions = [(input_lb, input_ub)]

    initial_splits: List[
        Tuple[
            Tensor,
            Tensor,
            Tuple[Tensor, Tensor, Tensor],
            int,
            Optional[Sequence[Sequence[Tuple[int, int, float]]]],
        ]
    ] = [
        (
            input_region[0],
            input_region[1],
            (property_matrix, property_threshold, combination_matrix),
            domain_splitting.max_depth,
            None,
        )
        for input_region in initial_input_regions
    ]
    return initial_splits


def batch_splits(
    queue: List[
        Tuple[
            Tensor,
            Tensor,
            Tuple[Tensor, Tensor, Tensor],
            int,
            Optional[Sequence[Sequence[Tuple[int, int, float]]]],
        ]
    ],
    batch_size: int,
) -> Tuple[
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    List[Optional[Sequence[Sequence[Tuple[int, int, float]]]]],
]:
    elements = []
    query_count = queue[0][2][1].shape[-1]
    for _ in range(batch_size):
        if not queue[0][2][1].shape[-1] == query_count:
            break
        elements.append(queue.pop(0))
        if len(queue) == 0:
            break
    # elements = queue[:min(batch_size, len(queue))]
    # queue = queue[min(batch_size, len(queue)):]
    input_lb = torch.cat([element[0] for element in elements], 0)
    input_ub = torch.cat([element[1] for element in elements], 0)
    property_matrix = torch.cat([element[2][0] for element in elements], 0)
    property_threshold = torch.cat([element[2][1] for element in elements], 0)
    combination_matrix = torch.cat([element[2][2] for element in elements], 0)
    max_depth = torch.tensor(
        [element[3] for element in elements], dtype=torch.int, device=input_lb.device
    )
    properties_to_verify_batch = [element[4] for element in elements]

    solved_properties = (property_matrix == 0).all(2).all(0)
    verified = (
        (property_matrix == 0)
        .all(2)
        .__and__(property_threshold < 0)
        .to(combination_matrix.dtype)
    )
    and_properties_verified = (
        torch.einsum(
            "bij,bj -> bi",
            combination_matrix,
            verified,
        )
        >= 1
    )
    property_matrix = property_matrix[:, ~solved_properties]
    property_threshold = property_threshold[:, ~solved_properties]
    combination_matrix = combination_matrix[:, :, ~solved_properties][
        :, ~and_properties_verified.all(0)
    ]
    assert (combination_matrix.sum(2) > 0).all()

    return (
        input_lb,
        input_ub,
        property_matrix,
        property_threshold,
        combination_matrix,
        max_depth,
        properties_to_verify_batch,
    )


def split_input_regions(
    input_regions: List[Tuple[Tensor, Tensor]],
    dim: Union[int, List[int]] = 0,
    splits: Union[int, List[int]] = 2,
) -> List[Tuple[Tensor, Tensor]]:
    input_shape = input_regions[0][0].shape
    if isinstance(splits, int):
        di = splits
    else:
        di = splits.pop(0)
    if isinstance(dim, int):
        d = dim
    else:
        d = dim.pop(0)
    new_input_regions = []
    for specLB, specUB in input_regions:
        specLB = specLB.flatten(1)
        specUB = specUB.flatten(1)
        d_lb = specLB[:, d].clone()
        d_ub = specUB[:, d].clone()
        d_range = d_ub - d_lb
        d_step = d_range / di
        for i in range(di):
            specLB[:, d] = d_lb + i * d_step
            specUB[:, d] = d_lb + (i + 1) * d_step
            new_input_regions.append(
                (
                    specLB.clone().view(-1, *input_shape[1:]),
                    specUB.clone().view(-1, *input_shape[1:]),
                )
            )
            assert (specLB[:, d] >= d_lb - 1e-7).all()
            assert (specUB[:, d] <= d_ub + 1e-7).all()
    if isinstance(splits, list) and isinstance(dim, list):
        if len(splits) == 0 or len(dim) == 0:
            return new_input_regions
        return split_input_regions(new_input_regions, dim=dim, splits=splits)
    elif isinstance(splits, list) and dim + 1 < len(splits):  # type: ignore # inference does not recognise that we have an int for dim here
        return split_input_regions(new_input_regions, dim=dim + 1, splits=splits)  # type: ignore # inference does not recognise that we have an int for dim here
    else:
        return new_input_regions


def consolidate_input_regions(
    input_regions: List[Tuple[Tensor, Tensor]]
) -> Tuple[Tensor, Tensor]:
    input_lb = torch.stack([x[0] for x in input_regions], 0).min(0)[0]
    input_ub = torch.stack([x[1] for x in input_regions], 0).max(0)[0]
    return input_lb, input_ub


# def list_minimum(inputs: List[Tensor]) -> Tensor:
#     if len(inputs) == 0:
#         assert False, "List Minimum undefined for empty lists"
#     elif len(inputs) == 1:
#         return inputs[0]
#     elif len(inputs) == 2:
#         return torch.minimum(inputs[0], inputs[1])
#     else:
#         return list_minimum(
#             [
#                 torch.minimum(inputs[2 * i], inputs[2 * i + 1])
#                 for i in range(len(inputs) // 2)
#             ]
#             + ([] if len(inputs) % 2 == 0 else [inputs[-1]])
#         )


# def list_maximum(inputs: List[Tensor]) -> Tensor:
#     if len(inputs) == 1:
#         return inputs[0]
#     if len(inputs) == 2:
#         return torch.maximum(inputs[0], inputs[1])
#     else:
#         return list_maximum(
#             [
#                 torch.maximum(inputs[2 * i], inputs[2 * i + 1])
#                 for i in range(len(inputs) // 2)
#             ]
#             + ([] if len(inputs) % 2 == 0 else [inputs[-1]])
#         )


def tensor_reduce(
    fun: Callable[[Tensor, Tensor], Tensor], in_tens: Sequence[Tensor]
) -> Tensor:
    return functools.reduce(fun, in_tens)
