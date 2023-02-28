from __future__ import annotations

from typing import Iterator, Optional, Tuple, Union, overload

import numpy as np
import torch
from torch import Tensor

from src.utilities.dependence_sets import DependenceSets

QueryCoef = Union[
    Tensor, DependenceSets
]  # batch_size x num_queries x current_layer_shape...


def num_queries(query_coef: QueryCoef) -> int:
    return (
        query_coef.sets.shape[1]
        if isinstance(query_coef, DependenceSets)
        else query_coef.shape[1]
    )


def get_output_bound_initial_query_coef(
    dim: Tuple[int, ...],
    intermediate_bounds_to_recompute: Optional[Tensor],  # None means recompute all.
    use_dependence_sets: bool,  # = False
    batch_size: int,
    dtype: Optional[torch.dtype],
    device: torch.device,
) -> QueryCoef:
    """
    Returns coefficients for a query that bounds all outputs of a layer of shape "dim".
    The coefficients are repeated for each batch index.
    """
    if dtype is None:
        dtype = torch.get_default_dtype()

    initial_bound_coef: QueryCoef
    if use_dependence_sets:
        num_query_channels, num_queries_spatial = dim[0], np.prod(dim[1:])
        repeats = batch_size, 1, num_queries_spatial, 1, 1, 1
        sets_final_shape = (
            batch_size,
            num_query_channels * num_queries_spatial,
            num_query_channels,
            1,
            1,
        )  # [B, C*WH, C, 1, 1]
        sets = (
            (
                torch.eye(num_query_channels, device=device, dtype=dtype)
                .unsqueeze(-1)
                .unsqueeze(-1)
                .unsqueeze(0)
                .unsqueeze(2)
            )
            .repeat(repeats)
            .view(sets_final_shape)
        )
        spatial_idxs = torch.arange(num_queries_spatial).repeat(num_query_channels)
        initial_bound_coef = DependenceSets(
            sets=sets,
            spatial_idxs=spatial_idxs,
            input_dim=dim,
            cstride=1,
            cpadding=0,
        )

        if intermediate_bounds_to_recompute is not None:
            initial_bound_coef = filter_queries(
                initial_bound_coef, intermediate_bounds_to_recompute
            )
            #  Todo integrate this update directly into the coef creation
    else:
        if intermediate_bounds_to_recompute is None:
            batch_repeats = batch_size, *([1] * (len(dim) + 1))
            initial_bound_coef = (
                torch.eye(np.prod(dim), device=device).view(-1, *dim).unsqueeze(0)
            ).repeat(batch_repeats)
        else:
            n_unstable_neurons = int(intermediate_bounds_to_recompute.sum().item())
            initial_bound_coef = torch.zeros(
                (
                    batch_size,  # type: ignore[call-overload]
                    n_unstable_neurons,
                    int(np.prod(dim)),
                ),
                device=device,
            )
            initial_bound_coef[
                :,
                torch.arange(n_unstable_neurons),
                intermediate_bounds_to_recompute,
            ] = 1.0
            initial_bound_coef = initial_bound_coef.view(
                (batch_size, n_unstable_neurons, *dim)
            )
    return initial_bound_coef


def get_output_bound_initial_query_coef_iterator(
    dim: Tuple[int, ...],
    intermediate_bounds_to_recompute: Optional[Tensor],  # None means recompute all.
    use_dependence_sets: bool,  # = False
    batch_size: int,
    slice_size: Optional[int],
    dtype: Optional[torch.dtype],
    device: torch.device,
) -> Iterator[Tuple[int, int, QueryCoef]]:
    """
    Returns coefficients for a query that bounds all outputs of a layer of shape "dim".
    The coefficients are repeated for each batch index.
    """
    if dtype is None:
        dtype = torch.get_default_dtype()

    if intermediate_bounds_to_recompute is not None:
        n_unstable_neurons = int(intermediate_bounds_to_recompute.sum().item())
    else:
        n_unstable_neurons = np.prod(dim)  # Is this correct?

    # TODO @Mark is this correct?
    if use_dependence_sets:
        num_query_channels, num_queries_spatial = dim[0], np.prod(dim[1:])
        repeats = batch_size, 1, num_queries_spatial, 1, 1, 1
        sets_final_shape = (
            batch_size,
            num_query_channels * num_queries_spatial,
            num_query_channels,
            1,
            1,
        )  # [B, C*WH, C, 1, 1]
        sets = (
            (
                torch.eye(num_query_channels, device=device, dtype=dtype)
                .unsqueeze(-1)
                .unsqueeze(-1)
                .unsqueeze(0)
                .unsqueeze(2)
            )
            .repeat(repeats)
            .view(sets_final_shape)
        )
        spatial_idxs = torch.arange(num_queries_spatial, device=device).repeat(
            num_query_channels
        )
        initial_bound_coef = DependenceSets(
            sets=sets,
            spatial_idxs=spatial_idxs,
            input_dim=dim,
            cstride=1,
            cpadding=0,
        )

        if intermediate_bounds_to_recompute is not None:
            initial_bound_coef = filter_queries(
                initial_bound_coef, intermediate_bounds_to_recompute
            )
            #  Todo integrate this update directly into the coef creation

        yield 0, n_unstable_neurons, initial_bound_coef

    else:
        if intermediate_bounds_to_recompute is not None:
            non_zero_idx = torch.nonzero(intermediate_bounds_to_recompute).flatten()
        else:
            non_zero_idx = torch.ones((n_unstable_neurons,), device=device)
        if slice_size is None:
            slice_size = len(non_zero_idx)
        offset = 0
        while offset < n_unstable_neurons:
            slice_size = min(slice_size, n_unstable_neurons - offset)
            curr_coef_slice = torch.zeros(
                (
                    batch_size,  # type: ignore[call-overload]
                    slice_size,
                    int(np.prod(dim)),
                ),
                device=device,
            )
            curr_coef_slice[:, :, non_zero_idx[offset : offset + slice_size]] = (
                torch.eye(slice_size, device=device)
                .unsqueeze(0)
                .repeat(batch_size, 1, 1)
            )

            offset += slice_size

            curr_coef_slice = curr_coef_slice.view((batch_size, slice_size, *dim))
            yield offset - slice_size, offset, curr_coef_slice


@overload
def filter_queries(query_coef: Tensor, mask: Tensor) -> Tensor:
    ...


@overload
def filter_queries(query_coef: DependenceSets, mask: Tensor) -> DependenceSets:  # type: ignore[misc]  # pre-commit-hook mypy thinks Tensor is Any
    ...


def filter_queries(query_coef: QueryCoef, mask: Tensor) -> QueryCoef:
    assert mask.ndim == 1  # TODO: ok?
    if isinstance(query_coef, DependenceSets):
        assert mask.shape[0] == query_coef.sets.shape[1]
        assert mask.shape[0] == query_coef.spatial_idxs.shape[0]
        return DependenceSets(
            sets=query_coef.sets[:, mask],
            spatial_idxs=query_coef.spatial_idxs[mask],
            input_dim=query_coef.input_dim,
            cstride=query_coef.cstride,
            cpadding=query_coef.cpadding,
        )
    else:
        assert isinstance(query_coef, Tensor)
        assert mask.shape[0] == query_coef.shape[1]
        return query_coef[:, mask]
