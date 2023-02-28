from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, Mapping, Optional, OrderedDict, Sequence, Tuple

import torch
from torch import Tensor

from src.state.layer_bounds import LayerBounds, ReadonlyLayerBounds
from src.state.prima_constraints import PrimaConstraints
from src.state.split_state import ReadonlySplitState, SplitState
from src.state.tags import LayerTag, NodeTag
from src.utilities.custom_typing import implement_properties_as_fields

if TYPE_CHECKING:
    from src.abstract_layers.abstract_container_module import ActivationLayer

INFEASIBILITY_CHECK_TOLERANCE = 1e-5


def _get_infeasibility_mask_from_intermediate_bounds(
    intermediate_bounds: Mapping[LayerTag, Tuple[Tensor, Tensor]],
    batch_size: int,
    device: torch.device,
) -> Tensor:
    result = torch.zeros(batch_size, dtype=torch.bool, device=device)
    for lb, ub in intermediate_bounds.values():
        result |= (
            (lb > ub + INFEASIBILITY_CHECK_TOLERANCE).flatten(start_dim=1).any(dim=1)
        )
    return result


class ReadonlyConstraints(ABC):
    @property
    @abstractmethod
    def split_state(self) -> Optional[ReadonlySplitState]:
        pass

    @property
    @abstractmethod
    def layer_bounds(self) -> ReadonlyLayerBounds:
        pass

    @property
    @abstractmethod
    def is_infeasible(self) -> Tensor:
        pass  # one entry per batch element

    @property
    @abstractmethod
    def prima_constraints(self) -> Optional[PrimaConstraints]:
        pass  # it seems this is shared state (Readonly not applied)

    @property
    @abstractmethod
    def batch_size(self) -> int:
        pass

    @property
    @abstractmethod
    def device(self) -> torch.device:
        pass

    def without_prima(self) -> ReadonlyConstraints:  # TODO: why needed?
        return Constraints.create_readonly(
            split_state=self.split_state,
            layer_bounds=self.layer_bounds,
            prima_constraints=None,
            is_infeasible=self.is_infeasible,
            batch_size=self.batch_size,
            device=self.device,
        )

    def deep_copy_to(self, device: torch.device) -> Constraints:
        return Constraints(
            split_state=self.split_state.deep_copy_to(device)
            if self.split_state is not None
            else None,
            layer_bounds=self.layer_bounds.deep_copy_to(device),
            prima_constraints=self.prima_constraints.deep_copy_to(device)
            if self.prima_constraints is not None
            else None,  # TODO: this is shared state, seems a bit weird
            is_infeasible=self.is_infeasible,  # (copied in constructor)
            batch_size=self.batch_size,
            device=device,
        )

    def deep_copy_to_no_clone(self, device: torch.device) -> ReadonlyConstraints:
        if self.prima_constraints is not None:
            self.prima_constraints.move_to(
                device
            )  # TODO: this is shared state, seems a bit weird
        return Constraints.create_readonly(
            split_state=self.split_state.deep_copy_to_no_clone(device)
            if self.split_state is not None
            else None,
            layer_bounds=self.layer_bounds.deep_copy_to_no_clone(device),
            prima_constraints=self.prima_constraints,
            is_infeasible=self.is_infeasible,  # (copied in constructor)
            batch_size=self.batch_size,
            device=device,
        )

    @property
    def is_fully_split(self) -> bool:
        return self.split_state is not None and self.split_state.is_fully_split

    def split(
        self,
        node_to_split: NodeTag,
        recompute_intermediate_bounds_after_branching: bool,
        layer: ActivationLayer,
        device: torch.device,
    ) -> Tuple[
        Constraints, Constraints, Sequence[LayerTag], float
    ]:  # (negative_split, positive_split, intermediate_bounds_to_be_kept_fixed, split_point)
        assert (
            self.split_state is not None
        ), "can only split states with a valid SplitState"
        (
            split_state_for_negative_split,
            split_state_for_positive_split,
            split_point,
        ) = self.split_state.split(self.layer_bounds, node_to_split, layer, device)

        (
            layer_bounds_for_negative_split,
            layer_bounds_for_positive_split,
            intermediate_bounds_to_be_kept_fixed,
        ) = self.layer_bounds.split(
            node_to_split,
            split_point,
            recompute_intermediate_bounds_after_branching,
            device,
        )

        negative_split = Constraints(
            split_state=split_state_for_negative_split,
            layer_bounds=layer_bounds_for_negative_split,
            prima_constraints=self.prima_constraints,
            is_infeasible=self.is_infeasible,  # (copied in constructor)
            batch_size=self.batch_size,
            device=device,
        )
        positive_split = Constraints(
            split_state=split_state_for_positive_split,
            layer_bounds=layer_bounds_for_positive_split,
            prima_constraints=self.prima_constraints,
            is_infeasible=self.is_infeasible,  # (copied in constructor)
            batch_size=self.batch_size,
            device=device,
        )

        return (
            negative_split,
            positive_split,
            intermediate_bounds_to_be_kept_fixed,
            split_point,
        )

    # for VerificationSubproblemQueue. TODO: why needed?
    def get_prima_constraints(self) -> Optional[PrimaConstraints]:
        return self.prima_constraints

    @abstractmethod
    def set_prima_coefficients(
        self, prima_coefficients: Dict[LayerTag, Tuple[Tensor, Tensor, Tensor]]
    ) -> None:
        pass


@implement_properties_as_fields
class Constraints(ReadonlyConstraints):
    split_state: Optional[SplitState]
    layer_bounds: LayerBounds
    prima_constraints: Optional[PrimaConstraints]
    is_infeasible: Tensor  # (one entry per batch element)
    batch_size: int
    device: torch.device

    def __init__(
        self,
        split_state: Optional[SplitState],
        layer_bounds: LayerBounds,
        prima_constraints: Optional[PrimaConstraints],
        is_infeasible: Optional[Tensor],
        batch_size: int,
        device: torch.device,
    ):
        self.split_state = split_state
        self.layer_bounds = layer_bounds
        self.prima_constraints = prima_constraints

        if is_infeasible is None:
            self.is_infeasible = torch.zeros(
                batch_size, dtype=torch.bool, device=device
            )
        else:
            self.is_infeasible = is_infeasible.clone().detach()

        assert self.split_state is None or self.split_state.batch_size == batch_size
        assert self.layer_bounds.batch_size == batch_size
        assert (
            self.prima_constraints is None
            or self.prima_constraints.batch_size == batch_size
        )
        assert self.is_infeasible.shape == (batch_size,)
        self.batch_size = batch_size

        assert self.split_state is None or self.split_state.device == device
        assert self.layer_bounds.device == device
        assert self.prima_constraints is None or self.prima_constraints.device == device
        self.device = device

    @classmethod
    def create_readonly(
        cls,
        split_state: Optional[ReadonlySplitState],
        layer_bounds: ReadonlyLayerBounds,
        prima_constraints: Optional[
            PrimaConstraints
        ],  # it seems this is shared state (Readonly not applied)
        is_infeasible: Optional[Tensor],
        batch_size: int,
        device: torch.device,
    ) -> ReadonlyConstraints:
        assert split_state is None or isinstance(split_state, SplitState)
        assert isinstance(layer_bounds, LayerBounds)
        assert prima_constraints is None or isinstance(
            prima_constraints, PrimaConstraints
        )
        return cls(
            split_state,
            layer_bounds,
            prima_constraints,
            is_infeasible,
            batch_size,
            device,
        )

    @classmethod
    def create_default(
        cls,
        split_state: Optional[SplitState],
        optimize_prima: bool,
        batch_size: int,
        device: torch.device,
    ) -> Constraints:
        layer_bounds = LayerBounds.create_default(batch_size, device)
        prima_constraints = (
            PrimaConstraints.create_default(batch_size, device)
            if optimize_prima
            else None
        )
        return cls(
            split_state=split_state,
            layer_bounds=layer_bounds,
            prima_constraints=prima_constraints,
            is_infeasible=None,
            batch_size=batch_size,
            device=device,
        )

    def without_prima(self) -> Constraints:  # TODO: why needed?
        result = Constraints(
            split_state=self.split_state,
            layer_bounds=self.layer_bounds,
            prima_constraints=None,
            batch_size=self.batch_size,
            is_infeasible=self.is_infeasible,  # TODO: a bit ugly (this is copied)
            device=self.device,
        )
        result.is_infeasible = self.is_infeasible  # TODO: a bit ugly
        return result

    def move_to(self, device: torch.device) -> None:
        if self.device == device:
            return
        if self.split_state is not None:
            self.split_state.move_to(device)
        self.layer_bounds.move_to(device)
        if self.prima_constraints is not None:
            self.prima_constraints.move_to(device)
        self.device = device

    def update_is_infeasible(self, is_infeasible: Tensor) -> None:
        assert is_infeasible.shape == (self.batch_size,)
        self.is_infeasible |= is_infeasible

    def update_feasibility_from_constraints(self) -> None:
        is_infeasible = _get_infeasibility_mask_from_intermediate_bounds(
            self.layer_bounds.intermediate_bounds, self.batch_size, self.device
        )
        self.update_is_infeasible(is_infeasible)

    def update_split_constraints(
        self,
        relu_layers: Sequence[LayerTag],
        bounds: OrderedDict[LayerTag, Tuple[Tensor, Tensor]],
    ) -> None:
        if self.split_state is None or self.split_state.split_constraints is None:
            return
        for layer_id in relu_layers:
            if layer_id in bounds and layer_id in self.split_state.split_constraints:
                self.split_state.refine_split_constraints_for_relu(
                    layer_id, bounds[layer_id]
                )

    # for VerificationSubproblemQueue. TODO: why needed?
    def set_prima_coefficients(
        self, prima_coefficients: Dict[LayerTag, Tuple[Tensor, Tensor, Tensor]]
    ) -> None:
        assert self.prima_constraints is not None
        self.prima_constraints.set_prima_coefficients(prima_coefficients)
