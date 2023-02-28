from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from copy import deepcopy
from typing import List, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor

from src.state.tags import LayerTag, NodeTag
from src.utilities.custom_typing import implement_properties_as_fields
from src.utilities.tensor_management import (
    deep_copy,
    deep_copy_to,
    deep_copy_to_no_clone,
    move_to,
)


class ReadonlyLayerBounds(ABC):
    @property
    @abstractmethod
    def intermediate_layer_bounds_to_be_kept_fixed(
        self,
    ) -> Sequence[LayerTag]:  # uniform across batch
        pass

    @property
    @abstractmethod
    def intermediate_bounds(self) -> Mapping[LayerTag, Tuple[Tensor, Tensor]]:
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
    def fixed_intermediate_bounds(
        self,
    ) -> OrderedDict[LayerTag, Tuple[Tensor, Tensor]]:  # TODO: make this a Mapping?
        # TODO: cache?
        return OrderedDict(
            (layer_id, bounds)
            for layer_id, bounds in self.intermediate_bounds.items()
            if layer_id in self.intermediate_layer_bounds_to_be_kept_fixed
        )

    def deep_copy_to(self, device: torch.device) -> LayerBounds:
        assert isinstance(self.intermediate_bounds, OrderedDict)
        return LayerBounds(
            self.intermediate_layer_bounds_to_be_kept_fixed,  # (immutable)
            deep_copy_to(self.intermediate_bounds, device),
            self.batch_size,
            device,
        )

    def deep_copy_to_no_clone(self, device: torch.device) -> ReadonlyLayerBounds:
        return LayerBounds.create_readonly(
            self.intermediate_layer_bounds_to_be_kept_fixed,  # (immutable)
            deep_copy_to_no_clone(self.intermediate_bounds, device),
            self.batch_size,
            device,
        )

    def split(
        self,
        node_to_split: NodeTag,
        split_point: Optional[float],
        recompute_intermediate_bounds_after_branching: bool,
        device: torch.device,
    ) -> Tuple[LayerBounds, LayerBounds, Sequence[LayerTag]]:

        intermediate_layer_bounds_to_be_kept_fixed: List[LayerTag] = []

        # Note if we dont do this sharing we apparently run out of memory during the splitting process
        if recompute_intermediate_bounds_after_branching:
            for layer_id in self.intermediate_bounds:
                intermediate_layer_bounds_to_be_kept_fixed.append(layer_id)
                if layer_id == node_to_split.layer:
                    break
        else:
            intermediate_layer_bounds_to_be_kept_fixed = list(
                self.intermediate_bounds.keys()
            )

        layer_id = node_to_split.layer
        index_in_layer = 0, *node_to_split.index
        # assert (
        #    self.intermediate_bounds[layer_id][0][index_in_layer] < 0
        #    and self.intermediate_bounds[layer_id][1][index_in_layer] > 0
        # ), "Attempted to split a stable node."

        if split_point is None:  # ReLU
            split_point = 0.0

        intermediate_bounds_for_negative_split = deepcopy(self.intermediate_bounds)
        assert isinstance(intermediate_bounds_for_negative_split, OrderedDict)
        intermediate_bounds_for_positive_split = deepcopy(self.intermediate_bounds)
        assert isinstance(intermediate_bounds_for_positive_split, OrderedDict)

        layer_upper_bounds_for_negative_split = (
            intermediate_bounds_for_negative_split[layer_id][1].clone().detach()
        )
        layer_upper_bounds_for_negative_split[index_in_layer] = split_point
        intermediate_bounds_for_negative_split[layer_id] = (
            intermediate_bounds_for_negative_split[layer_id][0],
            layer_upper_bounds_for_negative_split,
        )

        layer_lower_bounds_for_positive_split = (
            intermediate_bounds_for_positive_split[layer_id][0].clone().detach()
        )
        layer_lower_bounds_for_positive_split[index_in_layer] = split_point
        intermediate_bounds_for_positive_split[layer_id] = (
            layer_lower_bounds_for_positive_split,
            intermediate_bounds_for_positive_split[layer_id][1],
        )

        negative_split = LayerBounds(
            intermediate_layer_bounds_to_be_kept_fixed=intermediate_layer_bounds_to_be_kept_fixed,
            intermediate_bounds=intermediate_bounds_for_negative_split,
            batch_size=self.batch_size,
            device=device,
        )
        positive_split = LayerBounds(
            intermediate_layer_bounds_to_be_kept_fixed=intermediate_layer_bounds_to_be_kept_fixed,
            intermediate_bounds=intermediate_bounds_for_positive_split,
            batch_size=self.batch_size,
            device=device,
        )

        return (
            negative_split,
            positive_split,
            intermediate_layer_bounds_to_be_kept_fixed,
        )


@implement_properties_as_fields
class LayerBounds(ReadonlyLayerBounds):
    intermediate_layer_bounds_to_be_kept_fixed: Sequence[
        LayerTag
    ]  # uniform across batch
    intermediate_bounds: OrderedDict[LayerTag, Tuple[Tensor, Tensor]]
    batch_size: int
    device: torch.device

    def __init__(
        self,
        intermediate_layer_bounds_to_be_kept_fixed: Sequence[LayerTag],
        intermediate_bounds: OrderedDict[LayerTag, Tuple[Tensor, Tensor]],
        batch_size: int,
        device: torch.device,
    ):
        self.intermediate_layer_bounds_to_be_kept_fixed = (
            intermediate_layer_bounds_to_be_kept_fixed
        )
        self.intermediate_bounds = intermediate_bounds
        self.batch_size = batch_size
        self.device = device

    @classmethod
    def create_readonly(
        cls,
        intermediate_layer_bounds_to_be_kept_fixed: Sequence[LayerTag],
        intermediate_bounds: Mapping[LayerTag, Tuple[Tensor, Tensor]],
        batch_size: int,
        device: torch.device,
    ) -> ReadonlyLayerBounds:
        assert isinstance(intermediate_bounds, OrderedDict)
        return LayerBounds(
            intermediate_layer_bounds_to_be_kept_fixed,  # (immutable)
            intermediate_bounds,
            batch_size,
            device,
        )

    @classmethod
    def create_default(
        cls,
        batch_size: int,
        device: torch.device,
    ) -> LayerBounds:
        intermediate_layer_bounds_to_be_kept_fixed: Sequence[LayerTag] = []
        intermediate_bounds: OrderedDict[
            LayerTag, Tuple[Tensor, Tensor]
        ] = OrderedDict()
        return cls(
            intermediate_layer_bounds_to_be_kept_fixed,
            intermediate_bounds,
            batch_size,
            device,
        )

    def move_to(self, device: torch.device) -> None:
        if self.device is device:
            return
        self.intermediate_bounds = move_to(self.intermediate_bounds, device)
        self.device = device

    def improve(
        self,
        new_intermediate_bounds: OrderedDict[
            LayerTag, Tuple[Tensor, Tensor]
        ],  # TODO: make this a LayerBounds as well?
    ) -> None:
        for layer_id in new_intermediate_bounds:
            if layer_id in self.intermediate_bounds:
                self.intermediate_bounds[layer_id] = (
                    torch.maximum(
                        self.intermediate_bounds[layer_id][0],
                        new_intermediate_bounds[layer_id][0],
                    ).detach(),
                    torch.minimum(
                        self.intermediate_bounds[layer_id][1],
                        new_intermediate_bounds[layer_id][1],
                    ).detach(),
                )
            else:
                self.intermediate_bounds[layer_id] = deep_copy(
                    new_intermediate_bounds[layer_id]
                )
