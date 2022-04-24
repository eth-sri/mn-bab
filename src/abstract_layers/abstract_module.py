from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from src.exceptions.invalid_bounds import InvalidBoundsError
from src.mn_bab_shape import MN_BaB_Shape

INFEASIBILITY_CHECK_TOLERANCE = 1e-5


class AbstractModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_bounds: Optional[Tuple[Tensor, Tensor]] = None
        self.output_dim: Tuple[int, ...]
        self.dependence_set_applicable: Optional[bool] = None
        self.dependence_set_block = True

    @classmethod
    def from_concrete_module(
        cls, module: nn.Module, input_dim: Tuple[int, ...], **kwargs: Any
    ) -> AbstractModule:
        raise NotImplementedError

    def get_activation_layer_ids(
        self, act_layer_ids: Optional[List[int]] = None
    ) -> List[int]:
        if act_layer_ids is None:
            act_layer_ids = []
        return act_layer_ids

    def update_input_bounds(self, input_bounds: Tuple[Tensor, Tensor]) -> None:
        lb, ub = input_bounds

        if self.input_bounds is None:
            self.input_bounds = (lb, ub)
        else:
            self.input_bounds = (
                torch.maximum(lb, self.input_bounds[0]),
                torch.minimum(ub, self.input_bounds[1]),
            )

        invalid_bounds_mask_in_batch = [
            (
                best_lbs_of_batch_element
                > best_ubs_of_batch_element + INFEASIBILITY_CHECK_TOLERANCE
            )
            .any()
            .item()
            for best_lbs_of_batch_element, best_ubs_of_batch_element in zip(lb, ub)
        ]
        if any(invalid_bounds_mask_in_batch):
            raise InvalidBoundsError(invalid_bounds_mask_in_batch)

    def reset_input_bounds(self) -> None:
        self.input_bounds = None

    def detach_input_bounds(self) -> None:
        if self.input_bounds is not None:
            lb, ub = self.input_bounds
            self.input_bounds = lb.detach(), ub.detach()

    def backsubstitute(self, abstract_shape: MN_BaB_Shape) -> MN_BaB_Shape:
        raise NotImplementedError

    def propagate_interval(
        self, interval: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError
