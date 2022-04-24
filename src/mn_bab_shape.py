from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from src.utilities.dependence_sets import DependenceSets


class MN_BaB_Shape:
    """
    An abstract shape that is defined by an affine lower and upper bound.

    Attributes:
        lb_coef: linear coefficient of the lower bound
        ub_coef: linear coefficient of the upper bound
        lb_bias: bias of the lower bound
        ub_bias: bias of the upper bound
    """

    def __init__(
        self,
        lb_coef: Union[Tensor, DependenceSets],
        ub_coef: Union[Tensor, DependenceSets],
        lb_bias: Optional[Tensor] = None,
        ub_bias: Optional[Tensor] = None,
        optimizable_parameters: Optional[Dict[str, Dict[int, Tensor]]] = None,
        carried_over_optimizable_parameters: Optional[
            Dict[int, Dict[str, Dict[int, Tensor]]]
        ] = None,
        prima_coefficients: Optional[Dict[int, Tuple[Tensor, Tensor, Tensor]]] = None,
        prima_hyperparamters: Optional[Dict[str, float]] = None,
        split_constraints: Optional[Dict[int, Tensor]] = None,
        invalid_bounds_mask_in_batch: Optional[Sequence[bool]] = None,
    ) -> None:
        if isinstance(lb_coef, DependenceSets):
            assert prima_coefficients is None or not prima_coefficients
        self.device = lb_coef.device
        self.batch_size = (
            lb_coef.batch_size
            if isinstance(lb_coef, DependenceSets)
            else lb_coef.shape[0]
        )
        self.lb_coef = lb_coef.clone()
        self.ub_coef = ub_coef.clone()
        self.lb_bias = (
            torch.tensor(0, device=self.device) if lb_bias is None else lb_bias.clone()
        )
        self.ub_bias = (
            torch.tensor(0, device=self.device) if ub_bias is None else ub_bias.clone()
        )
        self.optimizable_parameters = optimizable_parameters
        self.carried_over_optimizable_parameters = carried_over_optimizable_parameters
        if prima_coefficients is None:
            self.prima_coefficients = {}
        else:
            self.prima_coefficients = prima_coefficients
        self.prima_hyperparameters = prima_hyperparamters
        self.split_constraints = split_constraints
        if invalid_bounds_mask_in_batch is None:
            self.invalid_bounds_mask_in_batch: Sequence[bool] = [
                False
            ] * self.batch_size
        else:
            self.invalid_bounds_mask_in_batch = invalid_bounds_mask_in_batch

    def clone(self, full: bool = True) -> MN_BaB_Shape:
        if full:
            return MN_BaB_Shape(
                self.lb_coef,
                self.ub_coef,
                self.lb_bias,
                self.ub_bias,
                _clone(self.optimizable_parameters),
                _clone(self.carried_over_optimizable_parameters),
                _clone(self.prima_coefficients),
                _clone(self.prima_hyperparameters),
                _clone(self.split_constraints),
                _clone(self.invalid_bounds_mask_in_batch),
            )
        else:
            return MN_BaB_Shape(
                self.lb_coef,
                self.ub_coef,
                self.lb_bias,
                self.ub_bias,
                self.optimizable_parameters,
                self.carried_over_optimizable_parameters,
                self.prima_coefficients,
                self.prima_hyperparameters,
                self.split_constraints,
                self.invalid_bounds_mask_in_batch,
            )

    @classmethod
    def construct_to_bound_all_outputs(
        cls,
        device: torch.device,
        output_dim: Tuple[int, ...],
        batch_size: int = 1,
        carried_over_optimizable_parameters: Optional[
            Dict[int, Dict[str, Dict[int, Tensor]]]
        ] = None,
        prima_coefficients: Optional[Dict[int, Tuple[Tensor, Tensor, Tensor]]] = None,
        prima_hyperparamters: Optional[Dict[str, float]] = None,
        split_constraints: Optional[Dict[int, Tensor]] = None,
        invalid_bounds_mask_in_batch: Optional[Sequence[bool]] = None,
        use_dependence_sets: bool = False,
    ) -> MN_BaB_Shape:
        if use_dependence_sets:
            num_query_channels, num_queries_spatial = output_dim[0], np.prod(
                output_dim[1:]
            )
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
                    torch.eye(num_query_channels, device=device)
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
                cstride=1,
                cpadding=0,
            )
        else:
            batch_repeats = batch_size, *([1] * (len(output_dim) + 1))
            initial_bound_coef = (
                torch.eye(np.prod(output_dim), device=device)
                .view(-1, *output_dim)
                .unsqueeze(0)
            ).repeat(batch_repeats)

        initial_lb_coef = initial_bound_coef
        initial_ub_coef = initial_bound_coef

        return cls(
            initial_lb_coef,
            initial_ub_coef,
            carried_over_optimizable_parameters=carried_over_optimizable_parameters,
            prima_coefficients=None if use_dependence_sets else prima_coefficients,
            prima_hyperparamters=prima_hyperparamters,
            split_constraints=split_constraints,
            invalid_bounds_mask_in_batch=invalid_bounds_mask_in_batch,
        )

    @classmethod
    def construct_with_same_optimization_state_as(
        cls,
        abstract_shape: MN_BaB_Shape,
        lb_coef: Tensor,
        ub_coef: Tensor,
        lb_bias: Optional[Tensor] = None,
        ub_bias: Optional[Tensor] = None,
    ) -> MN_BaB_Shape:
        assert isinstance(lb_coef, Tensor)  # no DependenceSets
        return cls(
            lb_coef.to(abstract_shape.device),
            ub_coef.to(abstract_shape.device),
            lb_bias.to(abstract_shape.device) if lb_bias is not None else None,
            ub_bias.to(abstract_shape.device) if ub_bias is not None else None,
            abstract_shape.optimizable_parameters,
            abstract_shape.carried_over_optimizable_parameters,
            abstract_shape.prima_coefficients,
            abstract_shape.prima_hyperparameters,
            abstract_shape.split_constraints,
            abstract_shape.invalid_bounds_mask_in_batch,
        )

    def uses_dependence_sets(self) -> bool:
        return isinstance(self.lb_coef, DependenceSets)

    def assert_matches_unstable_queries_mask(self, unstable_queries: Tensor) -> None:
        if unstable_queries is None:
            return
        nb_unstable = unstable_queries.sum()
        assert self.lb_bias.numel() == 1 or self.lb_bias.shape[1] == nb_unstable
        assert self.ub_bias.numel() == 1 or self.ub_bias.shape[1] == nb_unstable
        nb_active_queries_lb = (
            self.lb_coef.sets.shape[1]
            if self.uses_dependence_sets()
            else self.lb_coef.shape[1]  # type: ignore
        )
        assert nb_active_queries_lb == nb_unstable
        nb_active_queries_ub = (
            self.ub_coef.sets.shape[1]
            if self.uses_dependence_sets()
            else self.ub_coef.shape[1]  # type: ignore
        )
        assert nb_active_queries_ub == nb_unstable
        if self.uses_dependence_sets():
            assert self.lb_coef.spatial_idxs.shape[0] == nb_unstable
            assert self.ub_coef.spatial_idxs.shape[0] == nb_unstable

    def update_bounds(
        self,
        lb_coef: Union[Tensor, DependenceSets],
        ub_coef: Union[Tensor, DependenceSets],
        lb_bias: Optional[Tensor] = None,
        ub_bias: Optional[Tensor] = None,
    ) -> None:
        if isinstance(lb_coef, Tensor):
            assert lb_coef.shape[0] == self.batch_size
        self.lb_coef = lb_coef
        self.ub_coef = ub_coef
        self.lb_bias = torch.tensor(0) if lb_bias is None else lb_bias
        self.ub_bias = torch.tensor(0) if ub_bias is None else ub_bias

    def concretize(self, input_lb: Tensor, input_ub: Tensor) -> Tuple[Tensor, Tensor]:
        if len(input_lb.shape) in [2, 4]:
            batch_repeats = input_lb.shape
        elif len(input_lb.shape) in [1, 3]:
            batch_repeats = self.batch_size, *(input_lb.shape)
        else:
            raise RuntimeError("Unexpected number of dimensions for concretization.")

        output_lb, output_ub = self._matmul_of_coef_and_interval(
            input_lb.expand(batch_repeats),
            input_ub.expand(batch_repeats),
        )
        output_lb += self.lb_bias
        output_ub += self.ub_bias

        assert len(output_lb.shape) == 2

        return output_lb, output_ub

    def get_input_corresponding_to_lower_bound(
        self, input_lb: Tensor, input_ub: Tensor
    ) -> Tensor:
        assert isinstance(self.lb_coef, Tensor)  # no DependenceSets
        return torch.where(self.lb_coef > 0, input_lb, input_ub).view(
            self.batch_size, *input_lb.shape
        )

    def set_optimizable_parameters(self, starting_layer_id: int) -> None:
        if self.carried_over_optimizable_parameters is not None:
            self.optimizable_parameters = (
                self.carried_over_optimizable_parameters.setdefault(
                    starting_layer_id, {}
                )
            )

    def get_parameters(
        self,
        parameter_key: str,
        layer_id: int,
        parameter_shape: Tuple[int, ...],
        default_parameters: Optional[Tensor] = None,
    ) -> Tensor:
        assert self.optimizable_parameters is not None

        parameters_per_layer = self.optimizable_parameters.setdefault(parameter_key, {})
        if layer_id not in parameters_per_layer:
            if default_parameters is None:
                default_parameters = torch.zeros(*parameter_shape)
            default_parameters = default_parameters.to(self.device)
            parameters_per_layer[layer_id] = default_parameters

        requested_parameters = parameters_per_layer[layer_id]
        if not requested_parameters.requires_grad:
            requested_parameters.requires_grad_()
        return requested_parameters

    def refine_split_constraints_for(
        self, layer_id: int, bounds: Tuple[Tensor, Tensor]
    ) -> None:
        assert self.split_constraints
        input_lb, input_ub = bounds

        not_already_split_nodes = self.split_constraints[layer_id] == 0
        stable_inactive_nodes = input_ub <= 0
        stable_active_nodes = input_lb >= 0

        self.split_constraints[layer_id] = torch.where(
            (stable_inactive_nodes & not_already_split_nodes),
            torch.tensor(1, dtype=torch.int8, device=self.device),
            self.split_constraints[layer_id],
        )
        self.split_constraints[layer_id] = torch.where(
            (stable_active_nodes & not_already_split_nodes),
            torch.tensor(-1, dtype=torch.int8, device=self.device),
            self.split_constraints[layer_id],
        )

    def update_invalid_bounds_mask_in_batch(
        self, invalid_bounds_mask_in_batch: Sequence[bool]
    ) -> None:
        self.invalid_bounds_mask_in_batch = [
            invalid_before or invalid_now
            for invalid_before, invalid_now in zip(
                self.invalid_bounds_mask_in_batch, invalid_bounds_mask_in_batch
            )
        ]

    def filter_out_stable_queries(self, unstable_queries: Tensor) -> None:
        if self.uses_dependence_sets():
            assert unstable_queries.shape[0] == self.lb_coef.sets.shape[1]
            assert unstable_queries.shape[0] == self.lb_coef.spatial_idxs.shape[0]
            filtered_lb_coef = DependenceSets(
                sets=self.lb_coef.sets[:, unstable_queries],
                spatial_idxs=self.lb_coef.spatial_idxs[unstable_queries],
                cstride=self.lb_coef.cstride,
                cpadding=self.lb_coef.cpadding,
            )
            assert unstable_queries.shape[0] == self.ub_coef.sets.shape[1]
            assert unstable_queries.shape[0] == self.ub_coef.spatial_idxs.shape[0]
            filtered_ub_coef = DependenceSets(
                sets=self.ub_coef.sets[:, unstable_queries],
                spatial_idxs=self.ub_coef.spatial_idxs[unstable_queries],
                cstride=self.ub_coef.cstride,
                cpadding=self.ub_coef.cpadding,
            )
        else:
            assert unstable_queries.shape[0] == self.lb_coef.shape[1]  # type: ignore
            filtered_lb_coef = self.lb_coef[:, unstable_queries]  # type: ignore
            assert unstable_queries.shape[0] == self.ub_coef.shape[1]  # type: ignore
            filtered_ub_coef = self.ub_coef[:, unstable_queries]  # type: ignore

        filtered_lb_bias = (
            self.lb_bias[:, unstable_queries] if self.lb_bias.ndim > 0 else self.lb_bias
        )
        filtered_ub_bias = (
            self.ub_bias[:, unstable_queries] if self.ub_bias.ndim > 0 else self.ub_bias
        )

        self.update_bounds(
            filtered_lb_coef, filtered_ub_coef, filtered_lb_bias, filtered_ub_bias
        )

    def _elementwise_mul_of_coef_and_interval(
        self, interval_lb: Tensor, interval_ub: Tensor
    ) -> Tuple[Tensor, Tensor]:
        if self.uses_dependence_sets():
            interval_lb_matched = DependenceSets.unfold_to(interval_lb, self.lb_coef)
            interval_ub_matched = DependenceSets.unfold_to(interval_ub, self.ub_coef)
            lb, ub = self.lb_coef.sets, self.ub_coef.sets
        else:
            interval_lb_matched = interval_lb.unsqueeze(1)
            interval_ub_matched = interval_ub.unsqueeze(1)
            lb, ub = self.lb_coef, self.ub_coef

        result_lb = lb * torch.where(
            lb >= 0,
            interval_lb_matched,
            interval_ub_matched,
        )

        result_ub = ub * torch.where(
            ub >= 0,
            interval_ub_matched,
            interval_lb_matched,
        )

        return result_lb, result_ub

    def _matmul_of_coef_and_interval(
        self, interval_lb: Tensor, interval_ub: Tensor
    ) -> Tuple[Tensor, Tensor]:
        (
            elementwise_mul_lb,
            elementwise_mul_ub,
        ) = self._elementwise_mul_of_coef_and_interval(interval_lb, interval_ub)

        num_query_dimensions = 2

        elementwise_mul_lb = elementwise_mul_lb.view(
            *elementwise_mul_lb.shape[:num_query_dimensions], -1
        )

        elementwise_mul_ub = elementwise_mul_ub.view(
            *elementwise_mul_ub.shape[:num_query_dimensions], -1
        )

        return elementwise_mul_lb.sum(-1), elementwise_mul_ub.sum(-1)


def _clone(obj: Any) -> Any:
    if torch.is_tensor(obj):
        return obj.clone()
    elif isinstance(obj, dict):
        res: Any = {}
        for k, v in obj.items():
            res[k] = _clone(v)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(_clone(v))
        return res
    elif isinstance(obj, tuple):
        res = []
        for v in obj:
            res.append(_clone(v))
        return tuple(res)
    else:
        raise TypeError("Invalid type for clone")
