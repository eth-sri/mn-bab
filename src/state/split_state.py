from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import TYPE_CHECKING, Dict, Mapping, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from src.abstract_layers.abstract_container_module import ActivationLayer

import torch
from torch import Tensor

from src.state.layer_bounds import ReadonlyLayerBounds
from src.state.tags import LayerTag, NodeTag
from src.utilities.custom_typing import implement_properties_as_fields
from src.utilities.tensor_management import deep_copy_to, deep_copy_to_no_clone, move_to


class ReadonlySplitState(ABC):
    @property
    @abstractmethod
    def split_constraints(self) -> Mapping[LayerTag, Tensor]:  # TODO: int -> LayerTag
        pass

    @property
    @abstractmethod
    def split_points(self) -> Mapping[LayerTag, Tensor]:  # TODO: int -> LayerTag
        pass

    @property
    @abstractmethod
    def number_of_nodes_split(self) -> Sequence[int]:
        pass

    @property
    @abstractmethod
    def is_fully_split(self) -> bool:
        pass

    @property
    @abstractmethod
    def batch_size(self) -> int:
        pass

    @property
    @abstractmethod
    def device(self) -> torch.device:
        pass

    def unstable_node_mask_in_layer(
        self, layer_id: LayerTag
    ) -> Tensor:  # TODO: does this only work for ReLU?
        return (self.split_constraints[layer_id] == 0).detach()

    def deep_copy_to(self, device: torch.device) -> SplitState:
        assert isinstance(self.split_constraints, dict)
        assert isinstance(self.split_points, dict)
        return SplitState(
            deep_copy_to(self.split_constraints, device),
            deep_copy_to(self.split_points, device),
            self.number_of_nodes_split,  # (immutable)
            self.batch_size,
            device,
        )

    def deep_copy_to_no_clone(self, device: torch.device) -> ReadonlySplitState:
        return SplitState.create_readonly(
            deep_copy_to_no_clone(self.split_constraints, device),
            deep_copy_to_no_clone(self.split_points, device),
            self.number_of_nodes_split,  # (immutable)
            self.batch_size,
            device,
        )

    def split(
        self,
        bounds: ReadonlyLayerBounds,
        node_to_split: NodeTag,
        layer: ActivationLayer,
        device: torch.device,
    ) -> Tuple[
        SplitState,  # negative_split
        SplitState,  # positive_split
        float,  # split_point
    ]:
        layer_id = node_to_split.layer
        index_in_layer = 0, *node_to_split.index
        assert (
            self.split_constraints[layer_id][index_in_layer] == 0
        ), "Attempted to split a node that is already split."

        negative_split_constraints = deepcopy(self.split_constraints)
        assert isinstance(negative_split_constraints, dict)
        positive_split_constraints = deepcopy(self.split_constraints)
        assert isinstance(positive_split_constraints, dict)
        negative_split_points = deepcopy(self.split_points)
        assert isinstance(negative_split_points, dict)
        positive_split_points = deepcopy(self.split_points)
        assert isinstance(positive_split_points, dict)

        layer_split_constraints = self.split_constraints[layer_id]
        assert layer_split_constraints.shape[0] == 1

        neg_layer_split_constraints = layer_split_constraints.clone()
        neg_layer_split_constraints[index_in_layer] = 1
        negative_split_constraints[layer_id] = neg_layer_split_constraints

        pos_layer_split_constraints = layer_split_constraints.clone()
        pos_layer_split_constraints[index_in_layer] = -1
        positive_split_constraints[layer_id] = pos_layer_split_constraints

        from src.abstract_layers.abstract_relu import ReLU

        if isinstance(layer, ReLU):
            split_point = torch.Tensor([0.0])
        else:
            assert layer_id in self.split_points
            layer_split_points = self.split_points[layer_id]
            neg_layer_split_points = layer_split_points.clone()
            pos_layer_split_points = layer_split_points.clone()
            lbs, ubs = bounds.intermediate_bounds[layer_id]
            lb, ub = lbs[index_in_layer], ubs[index_in_layer]
            split_point = layer.get_split_points(lb, ub)  # type: ignore  # "get_split_points" is a classmethod not Tensor

            neg_layer_split_points[index_in_layer] = split_point.item()
            pos_layer_split_points[index_in_layer] = split_point.item()
            negative_split_points[layer_id] = neg_layer_split_points
            positive_split_points[layer_id] = pos_layer_split_points

        number_of_nodes_split: Sequence[int] = [
            n + 1 for n in self.number_of_nodes_split
        ]
        negative_split = SplitState(
            negative_split_constraints,
            negative_split_points,
            number_of_nodes_split,
            self.batch_size,
            device,
        )
        positive_split = SplitState(
            positive_split_constraints,
            positive_split_points,
            number_of_nodes_split,
            self.batch_size,
            device,
        )

        return (
            negative_split,
            positive_split,
            split_point.item(),
        )


@implement_properties_as_fields
class SplitState(
    ReadonlySplitState
):  # TODO: probably this class should know which layers are ReLUs
    split_constraints: Dict[LayerTag, Tensor]
    split_points: Dict[LayerTag, Tensor]
    number_of_nodes_split: Sequence[int]  # (for multiple batches)
    is_fully_split: bool
    batch_size: int
    device: torch.device

    def __init__(
        self,
        split_constraints: Dict[LayerTag, Tensor],
        split_points: Dict[LayerTag, Tensor],
        number_of_nodes_split: Sequence[int],
        batch_size: int,
        device: torch.device,
    ):
        self.split_constraints = split_constraints
        self.split_points = split_points
        self.number_of_nodes_split = number_of_nodes_split
        self.is_fully_split = all(
            [(splits != 0).all() for splits in split_constraints.values()]
        )

        assert len(number_of_nodes_split) == batch_size
        self.batch_size = batch_size
        self.device = device

    @classmethod
    def create_readonly(
        cls,
        split_constraints: Mapping[int, Tensor],
        split_points: Mapping[int, Tensor],
        number_of_nodes_split: Sequence[int],
        batch_size: int,
        device: torch.device,
    ) -> ReadonlySplitState:
        assert isinstance(split_constraints, dict)
        assert isinstance(split_points, dict)
        return cls(
            split_constraints,
            split_points,
            number_of_nodes_split,  # (immutable)
            batch_size,
            device,
        )

    @classmethod
    def create_default(
        cls,
        split_constraints: Optional[Dict[LayerTag, Tensor]],
        split_points: Optional[Dict[LayerTag, Tensor]],
        batch_size: int,
        device: torch.device,
    ) -> SplitState:
        if split_constraints is None:
            split_constraints = {}
        if split_points is None:
            split_points = {}
        number_of_nodes_split = [0]
        return cls(
            split_constraints, split_points, number_of_nodes_split, batch_size, device
        )

    def move_to(self, device: torch.device) -> None:
        if self.device is device:
            return
        self.split_constraints = move_to(self.split_constraints, device)
        self.split_points = move_to(self.split_points, device)
        self.device = device

    def refine_split_constraints_for_relu(
        self, layer_id: LayerTag, bounds: Tuple[Tensor, Tensor]
    ) -> None:
        assert layer_id in self.split_constraints
        assert layer_id not in self.split_points
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

    # def refine_split_constraints_for_sig( # Sigmoid / Tanh
    #     self, layer_id: int, bounds: Tuple[Tensor, Tensor]
    # ) -> None:
    #     assert layer_id in self.split_points.keys()
    #     input_lb, input_ub = bounds
    #     not_already_split_nodes = self.split_constraints[layer_id] == 0

    #     # Resets our splitting in case the bounds move across the split point
    #     split_points = self.split_points[layer_id]
    #     no_longer_split_nodes = (input_lb >= split_points) | (
    #         input_ub <= split_points
    #     )
    #     self.split_constraints[layer_id] = torch.where(
    #         (~not_already_split_nodes & no_longer_split_nodes),
    #         torch.tensor(0, dtype=torch.int8, device=self.device),
    #         self.split_constraints[layer_id],
    #     )
