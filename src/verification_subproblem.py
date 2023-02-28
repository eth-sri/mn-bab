from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch
from torch import Tensor

from src.state.constraints import PrimaConstraints
from src.state.split_state import SplitState
from src.state.subproblem_state import ReadonlySubproblemState, SubproblemState
from src.state.tags import LayerTag, NodeTag
from src.utilities.custom_typing import implement_properties_as_fields

if TYPE_CHECKING:
    from src.abstract_layers.abstract_container_module import ActivationLayer


class ReadonlyVerificationSubproblem(ABC):
    @property
    @abstractmethod
    def lower_bound(self) -> float:
        pass

    @property
    @abstractmethod
    def upper_bound(self) -> float:
        pass

    @property
    @abstractmethod
    def is_infeasible(self) -> bool:
        pass

    @property
    @abstractmethod
    def subproblem_state(self) -> ReadonlySubproblemState:
        pass

    @property
    @abstractmethod
    def device(self) -> torch.device:
        pass

    def deep_copy_to(self, device: torch.device) -> VerificationSubproblem:
        return VerificationSubproblem(
            self.lower_bound,
            self.upper_bound,
            self.subproblem_state.deep_copy_to(device),
            device,
        )

    def deep_copy_to_no_clone(
        self, device: torch.device
    ) -> ReadonlyVerificationSubproblem:
        return VerificationSubproblem.create_readonly(
            self.lower_bound,
            self.upper_bound,
            self.subproblem_state.deep_copy_to_no_clone(device),
            device,
        )

    def split(
        self,
        node_to_split: NodeTag,
        recompute_intermediate_bounds_after_branching: bool,
        layer: ActivationLayer,
    ) -> Tuple[
        ReadonlyVerificationSubproblem, ReadonlyVerificationSubproblem
    ]:  # readonly because the resulting parameters alias the original ones

        (
            subproblem_state_for_negative_split,
            subproblem_state_for_positive_split,
        ) = self.subproblem_state.split(
            node_to_split,
            recompute_intermediate_bounds_after_branching,
            layer,
            self.device,
        )
        negative_split_subproblem = VerificationSubproblem.create_readonly(
            self.lower_bound,
            self.upper_bound,
            subproblem_state_for_negative_split,
            self.device,
        )
        positive_split_subproblem = VerificationSubproblem.create_readonly(
            self.lower_bound,
            self.upper_bound,
            subproblem_state_for_positive_split,
            self.device,
        )

        return negative_split_subproblem, positive_split_subproblem

    @property
    def is_fully_split(self) -> bool:
        return self.subproblem_state.is_fully_split

    # for VerificationSubproblemQueue. TODO: why needed?
    def get_layer_id_to_index(
        self,
    ) -> Dict[int, int]:  # TODO: this seems out of place here
        return self.subproblem_state.get_layer_id_to_index()

    # for VerificationSubproblemQueue. TODO: why needed?
    def set_prima_coefficients(
        self, prima_coefficients: Dict[LayerTag, Tuple[Tensor, Tensor, Tensor]]
    ) -> None:  # TODO: get rid of this?
        self.subproblem_state.set_prima_coefficients(prima_coefficients)

    def get_prima_constraints(self) -> Optional[PrimaConstraints]:
        return self.subproblem_state.get_prima_constraints()


@implement_properties_as_fields
class VerificationSubproblem(ReadonlyVerificationSubproblem):
    """
    Bounds results of a batch of queries on a subproblem:
    lower_bound <= min_x query_i*network(x) <= upper_bound for all i, where
    the minimum is w.r.t additional Constraints stored in subproblem_state.

    subproblem_state also contains Parameters that are a witness for
    the given lower bound. The upper bound is typically determined using a
    concrete evaluation of the neural network on an input derived from the constraints.

    (The coefficients of queries are currently stored and batched separately.)
    """

    lower_bound: float  # TODO: add lower_bounds: Sequence[float] to support batching nicely?
    upper_bound: float  # TODO: add upper_bounds: Sequence[float] to support batching nicely?
    subproblem_state: SubproblemState  # TODO: proper support for batching
    device: torch.device

    @property
    def is_infeasible(self) -> bool:  # TODO: proper support for batching?
        return all(self.subproblem_state.is_infeasible)

    def __init__(
        self,
        lower_bound: float,
        upper_bound: float,
        subproblem_state: SubproblemState,
        device: torch.device,
    ):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.subproblem_state = subproblem_state  # TODO: proper support for batching

        self.device = device
        self.subproblem_state.move_to(device)

    @classmethod
    def create_readonly(
        cls,
        lower_bound: float,
        upper_bound: float,
        subproblem_state: ReadonlySubproblemState,
        device: torch.device,
    ) -> ReadonlyVerificationSubproblem:
        assert isinstance(subproblem_state, SubproblemState)
        return cls(lower_bound, upper_bound, subproblem_state, device)

    @classmethod
    def create_default(
        cls,
        lower_bound: float,
        upper_bound: float,
        split_state: Optional[SplitState],
        optimize_prima: bool,
        device: torch.device,
    ) -> VerificationSubproblem:
        subproblem_state = SubproblemState.create_default(
            split_state=split_state,
            optimize_prima=optimize_prima,
            batch_size=1,
            device=device,
            use_params=True,
        )
        return cls(
            lower_bound,
            upper_bound,
            subproblem_state,
            device,
        )

    def move_to(self, device: torch.device) -> None:
        if self.device == device:
            return
        self.subproblem_state.move_to(device)
        self.device = device
