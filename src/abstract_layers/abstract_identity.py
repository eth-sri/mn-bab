from __future__ import annotations

from typing import Any, Optional, Tuple

import torch.nn as nn
from torch import Tensor

from src.abstract_domains.ai_util import AbstractElement
from src.abstract_layers.abstract_module import AbstractModule
from src.mn_bab_shape import MN_BaB_Shape
from src.utilities.config import BacksubstitutionConfig
from src.verification_subproblem import SubproblemState


class Identity(nn.Identity, AbstractModule):
    def __init__(self, input_dim: Tuple[int, ...]) -> None:
        super(Identity, self).__init__()
        self.output_dim = input_dim
        self.dependence_set_block = False

    @classmethod
    def from_concrete_module(  # type: ignore[override] # (checked at runtime)
        cls, module: nn.Identity, input_dim: Tuple[int, ...], **kwargs: Any
    ) -> Identity:
        assert isinstance(module, nn.Identity)
        return cls(input_dim)

    def backsubstitute(
        self, config: BacksubstitutionConfig, abstract_shape: MN_BaB_Shape
    ) -> MN_BaB_Shape:
        return abstract_shape

    def propagate_interval(
        self,
        interval: Tuple[Tensor, Tensor],
        use_existing_bounds: Optional[bool] = None,
        subproblem_state: Optional[SubproblemState] = None,
        activation_layer_only: bool = False,
        set_input: bool = True,
        set_output: bool = True,
    ) -> Tuple[Tensor, Tensor]:
        return interval

    def propagate_abstract_element(
        self,
        abs_input: AbstractElement,
        use_existing_bounds: Optional[bool] = None,
        activation_layer_only: bool = False,
        set_input: bool = True,
        set_output: bool = True,
    ) -> AbstractElement:
        return abs_input
