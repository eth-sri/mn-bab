from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from src.abstract_domains.ai_util import AbstractElement
from src.exceptions.invalid_bounds import InvalidBoundsError
from src.mn_bab_shape import MN_BaB_Shape
from src.state.constraints import INFEASIBILITY_CHECK_TOLERANCE
from src.state.tags import LayerTag
from src.utilities.config import BacksubstitutionConfig
from src.verification_subproblem import SubproblemState


class AbstractModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_bounds: Optional[Tuple[Tensor, Tensor]] = None
        self.optim_input_bounds: Optional[Tuple[Tensor, Tensor]] = None
        self.output_bounds: Optional[Tuple[Tensor, Tensor]] = None
        self.output_dim: Tuple[int, ...]
        self.dependence_set_applicable: Optional[bool] = None
        self.dependence_set_block = True

    @classmethod
    def from_concrete_module(
        cls, module: nn.Module, input_dim: Tuple[int, ...], **kwargs: Any
    ) -> AbstractModule:
        raise NotImplementedError

    def get_activation_layer_ids(
        self, act_layer_ids: Optional[List[LayerTag]] = None
    ) -> List[LayerTag]:
        if act_layer_ids is None:
            act_layer_ids = []
        return act_layer_ids

    def get_relu_layer_ids(
        self, act_layer_ids: Optional[List[LayerTag]] = None
    ) -> List[LayerTag]:
        if act_layer_ids is None:
            act_layer_ids = []
        return act_layer_ids

    def update_input_bounds(
        self, input_bounds: Tuple[Tensor, Tensor], check_feasibility: bool = True
    ) -> None:
        lb, ub = input_bounds

        if self.input_bounds is None:
            self.input_bounds = (lb, ub)
        else:
            self.input_bounds = (
                torch.maximum(lb, self.input_bounds[0].view_as(lb)),
                torch.minimum(ub, self.input_bounds[1].view_as(ub)),
            )

        if check_feasibility:
            invalid_bounds_mask_in_batch = (
                (
                    self.input_bounds[0]
                    > self.input_bounds[1] + INFEASIBILITY_CHECK_TOLERANCE
                )
                .flatten(start_dim=1)
                .any(dim=1)
            )

            if invalid_bounds_mask_in_batch.any():
                raise InvalidBoundsError(invalid_bounds_mask_in_batch)

    def reset_input_bounds(self) -> None:
        self.input_bounds = None

    def reset_optim_input_bounds(self) -> None:
        self.optim_input_bounds = None

    def detach_input_bounds(self) -> None:
        if self.input_bounds is not None:
            lb, ub = self.input_bounds
            self.input_bounds = lb.detach(), ub.detach()

    def update_output_bounds(self, output_bounds: Tuple[Tensor, Tensor]) -> None:
        lb, ub = output_bounds

        if self.output_bounds is None:
            self.output_bounds = (lb, ub)
        else:
            self.output_bounds = (
                torch.maximum(lb, self.output_bounds[0]),
                torch.minimum(ub, self.output_bounds[1]),
            )

    def reset_output_bounds(self) -> None:
        self.output_bounds = None

    def detach_output_bounds(self) -> None:
        if self.output_bounds is not None:
            lb, ub = self.output_bounds
            self.output_bounds = lb.detach(), ub.detach()

    def backsubstitute(
        self, config: BacksubstitutionConfig, abstract_shape: MN_BaB_Shape
    ) -> MN_BaB_Shape:
        raise NotImplementedError

    def propagate_interval(
        self,
        interval: Tuple[Tensor, Tensor],
        use_existing_bounds: Optional[bool] = None,
        subproblem_state: Optional[SubproblemState] = None,
        activation_layer_only: bool = False,
        set_input: bool = True,
        set_output: bool = True,
    ) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError

    def propagate_abstract_element(
        self,
        abs_input: AbstractElement,
        use_existing_bounds: Optional[bool] = None,
        activation_layer_only: bool = False,
        set_input: bool = True,
        set_output: bool = True,
    ) -> AbstractElement:
        raise NotImplementedError
