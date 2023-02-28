from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch
from torch import Tensor

from src.state.constraints import Constraints, ReadonlyConstraints
from src.state.parameters import Parameters, ReadonlyParameters
from src.state.prima_constraints import PrimaConstraints
from src.state.split_state import SplitState
from src.state.tags import LayerTag, NodeTag
from src.utilities.custom_typing import implement_properties_as_fields

if TYPE_CHECKING:
    from src.abstract_layers.abstract_container_module import ActivationLayer


class ReadonlySubproblemState(ABC):
    @property
    @abstractmethod
    def constraints(self) -> ReadonlyConstraints:
        pass

    @property
    @abstractmethod
    def parameters(self) -> ReadonlyParameters:
        pass

    @property
    @abstractmethod
    def batch_size(self) -> int:
        pass

    @property
    @abstractmethod
    def device(self) -> torch.device:
        pass

    @property
    def is_infeasible(self) -> Tensor:
        return self.constraints.is_infeasible

    def without_prima(self) -> ReadonlySubproblemState:
        return SubproblemState.create_readonly(
            self.constraints.without_prima(),
            self.parameters,
            self.batch_size,
            self.device,
        )

    def with_new_parameters(self) -> ReadonlySubproblemState:  # TODO: why necessary?
        return SubproblemState.create_readonly(
            self.constraints,
            Parameters.create_default(self.batch_size, self.device, use_params=True),
            self.batch_size,
            self.device,
        )

    def without_parameters(
        self,
    ) -> ReadonlySubproblemState:  # TODO: probably it would be better to put use_params into the backsubtitution config.
        return SubproblemState.create_readonly(
            self.constraints,
            Parameters.create_default(self.batch_size, self.device, use_params=False),
            self.batch_size,
            self.device,
        )

    def deep_copy_to(self, device: torch.device) -> SubproblemState:
        return SubproblemState(
            self.constraints.deep_copy_to(device),
            self.parameters.deep_copy_to(device),
            self.batch_size,
            device,
        )

    def deep_copy_to_no_clone(self, device: torch.device) -> ReadonlySubproblemState:
        return SubproblemState.create_readonly(
            self.constraints.deep_copy_to_no_clone(device),
            self.parameters.deep_copy_to_no_clone(device),
            self.batch_size,
            device,
        )

    def split(
        self,
        node_to_split: NodeTag,
        recompute_intermediate_bounds_after_branching: bool,
        layer: ActivationLayer,
        device: torch.device,
    ) -> Tuple[
        ReadonlySubproblemState, ReadonlySubproblemState
    ]:  # readonly because the resulting parameters alias the original ones
        (
            constraints_for_negative_split,
            constraints_for_positive_split,
            intermediate_layer_bounds_to_be_kept_fixed,
            split_point,
        ) = self.constraints.split(
            node_to_split, recompute_intermediate_bounds_after_branching, layer, device
        )

        active_parameters = self.parameters.get_active_parameters_after_split(
            recompute_intermediate_bounds_after_branching,
            intermediate_layer_bounds_to_be_kept_fixed,
            device,
        )

        negative_split = SubproblemState.create_readonly(
            constraints_for_negative_split, active_parameters, self.batch_size, device
        )
        positive_split = SubproblemState.create_readonly(
            constraints_for_positive_split, active_parameters, self.batch_size, device
        )

        return (negative_split, positive_split)

    @property
    def is_fully_split(self) -> bool:
        return self.constraints.is_fully_split

    def get_layer_id_to_index(
        self,
    ) -> Dict[int, int]:  # TODO: this seems out of place here
        layer_id_to_index: Dict[int, int] = {}
        assert (
            self.constraints.split_state is not None
        ), "need valid split state for branch and bound"
        for index, layer_id in enumerate(
            self.constraints.split_state.split_constraints
        ):
            layer_id_to_index[layer_id] = index
        return layer_id_to_index

    # for VerificationSubproblemQueue. TODO: why needed?
    def get_prima_constraints(
        self,
    ) -> Optional[PrimaConstraints]:
        return self.constraints.get_prima_constraints()

    def set_prima_coefficients(
        self, prima_coefficients: Dict[LayerTag, Tuple[Tensor, Tensor, Tensor]]
    ) -> None:  # TODO: get rid of this?
        self.constraints.set_prima_coefficients(prima_coefficients)


@implement_properties_as_fields
class SubproblemState(ReadonlySubproblemState):
    """
    Represents an optimization problem min_x query_i*f(x) w.r.t constraints
    together with parameters that are used to bound it from below.

    (The coefficients of queries are currently stored and batched separately.)
    """

    constraints: Constraints
    parameters: Parameters
    batch_size: int
    device: torch.device

    def __init__(
        self,
        constraints: Constraints,
        parameters: Parameters,
        batch_size: int,
        device: torch.device,
    ):
        self.constraints = constraints
        self.parameters = parameters

        assert self.constraints.batch_size == batch_size
        assert self.parameters.batch_size == batch_size
        self.batch_size = batch_size

        assert self.constraints.device == device
        assert self.parameters.device == device
        self.device = device

    @classmethod
    def create_readonly(
        cls,
        constraints: ReadonlyConstraints,
        parameters: ReadonlyParameters,
        batch_size: int,
        device: torch.device,
    ) -> ReadonlySubproblemState:
        assert isinstance(constraints, Constraints)
        assert isinstance(parameters, Parameters)
        return cls(
            constraints,
            parameters,
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
        use_params: bool,  # TODO: probably it would be better to put use_params into the backsubtitution config.
    ) -> SubproblemState:
        constraints = Constraints.create_default(
            split_state, optimize_prima, batch_size, device
        )
        parameters = Parameters.create_default(batch_size, device, use_params)
        return cls(constraints, parameters, batch_size, device)

    def without_prima(self) -> SubproblemState:
        return SubproblemState(
            self.constraints.without_prima(),
            self.parameters,
            self.batch_size,
            self.device,
        )

    def with_new_parameters(self) -> SubproblemState:  # TODO: why necessary?
        return SubproblemState(
            self.constraints,
            Parameters.create_default(self.batch_size, self.device, use_params=True),
            self.batch_size,
            self.device,
        )

    def without_parameters(
        self,
    ) -> SubproblemState:  # TODO: probably it would be better to put use_params into the backsubtitution config.
        return SubproblemState(
            self.constraints,
            Parameters.create_default(self.batch_size, self.device, use_params=False),
            self.batch_size,
            self.device,
        )

    def move_to(self, device: torch.device) -> None:
        if self.device == device:
            return
        self.constraints.move_to(device)
        self.parameters.move_to(device)
        self.device = device

    def update_feasibility(self) -> None:
        self.constraints.update_feasibility_from_constraints()
