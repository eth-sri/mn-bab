from __future__ import annotations

from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor

from src.abstract_layers.abstract_max_pool2d import MaxPool2d
from src.abstract_layers.abstract_module import AbstractModule
from src.abstract_layers.abstract_relu import ReLU
from src.abstract_layers.abstract_sigmoid import Sigmoid
from src.abstract_layers.abstract_tanh import Tanh
from src.mn_bab_shape import MN_BaB_Shape
from src.state.tags import LayerTag
from src.utilities.config import BacksubstitutionConfig
from src.verification_subproblem import SplitState

ActivationLayer = Union[
    ReLU, Sigmoid, Tanh, MaxPool2d
]  # TODO: add common superclass for activation layers?

ActivationLayers = [ReLU, Sigmoid, Tanh, MaxPool2d]


class AbstractContainerModule(AbstractModule):
    def set_intermediate_input_bounds(
        self, intermediate_bounds: OrderedDict[LayerTag, Tuple[Tensor, Tensor]]
    ) -> None:
        raise NotImplementedError

    def get_default_split_constraints(
        self, batch_size: int, device: torch.device
    ) -> Dict[LayerTag, Tensor]:
        raise NotImplementedError

    def get_default_split_points(
        self, batch_size: int, device: torch.device
    ) -> Dict[LayerTag, Tensor]:
        raise NotImplementedError

    def get_default_split_state(
        self, batch_size: int, device: torch.device
    ) -> SplitState:
        return SplitState.create_default(
            split_constraints=self.get_default_split_constraints(batch_size, device),
            split_points=self.get_default_split_points(batch_size, device),
            batch_size=batch_size,
            device=device,
        )

    def get_activation_layers(self) -> Dict[LayerTag, ActivationLayer]:
        raise NotImplementedError

    def set_dependence_set_applicability(self, applicable: bool = True) -> None:
        raise NotImplementedError

    def get_current_intermediate_bounds(
        self,
    ) -> OrderedDict[LayerTag, Tuple[Tensor, Tensor]]:
        raise NotImplementedError

    def get_current_optimized_intermediate_bounds(
        self,
    ) -> OrderedDict[LayerTag, Tuple[Tensor, Tensor]]:
        raise NotImplementedError

    def backsubstitute_shape(
        self,
        config: BacksubstitutionConfig,
        input_lb: Tensor,
        input_ub: Tensor,
        abstract_shape: MN_BaB_Shape,
        from_layer_index: Optional[int],
        propagate_preceeding_callback: Optional[
            Callable[
                [BacksubstitutionConfig, MN_BaB_Shape, bool],
                Tuple[MN_BaB_Shape, Tuple[Tensor, Tensor]],
            ]
        ],
        preceeding_layers: Optional[List[Any]],
        use_early_termination_for_current_query: bool,  # = False,
        full_back_prop: bool,  # = False,
        optimize_intermediate_bounds: bool,  # = False,
    ) -> Tuple[MN_BaB_Shape, Tuple[Tensor, Tensor]]:
        raise NotImplementedError

    def forward_pass(
        self,
        config: BacksubstitutionConfig,
        input_lb: Tensor,
        input_ub: Tensor,
        propagate_preceeding_callback: Optional[
            Callable[
                [BacksubstitutionConfig, MN_BaB_Shape, bool],
                Tuple[MN_BaB_Shape, Tuple[Tensor, Tensor]],
            ]
        ],
        preceeding_layers: Optional[List[Any]],
        ibp_call: Callable[[], None],
        timeout: float,
    ) -> None:
        raise NotImplementedError
