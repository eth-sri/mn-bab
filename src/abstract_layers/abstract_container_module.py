from __future__ import annotations

from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple

from torch import Tensor

from src.abstract_layers.abstract_module import AbstractModule
from src.mn_bab_shape import MN_BaB_Shape


class AbstractContainerModule(AbstractModule):
    def set_intermediate_input_bounds(
        self, intermediate_bounds: OrderedDict[int, Tuple[Tensor, Tensor]]
    ) -> None:
        raise NotImplementedError

    def get_default_split_constraints(self) -> Dict[int, Tensor]:
        raise NotImplementedError

    def set_dependence_set_applicability(self, applicable: bool = True) -> None:
        raise NotImplementedError

    def get_current_intermediate_bounds(
        self,
    ) -> OrderedDict[int, Tuple[Tensor, Tensor]]:
        raise NotImplementedError

    def backsubstitute_shape(
        self,
        abstract_shape: MN_BaB_Shape,
        input_lb: Tensor,
        input_ub: Tensor,
        starting_layer_id: int,
        from_layer_index: Optional[int],
        layer_ids_for_which_to_compute_prima_constraints: List[int],
        best_intermediate_bounds_so_far: Optional[
            OrderedDict[int, Tuple[Tensor, Tensor]]
        ] = None,
        propagate_preceeding_callback: Optional[
            Callable[
                [MN_BaB_Shape, Optional[Tensor], bool, bool],
                Tuple[MN_BaB_Shape, Tuple[Tensor, Tensor], Optional[Tensor]],
            ]
        ] = None,
        preceeding_layers: Optional[List[Any]] = None,
        use_dependence_sets: bool = False,
        use_early_termination: bool = False,
        use_early_termination_for_current_query: bool = False,
        full_back_prop: bool = False,
        unstable_queries: Optional[Tensor] = None,
    ) -> Tuple[MN_BaB_Shape, Tuple[Tensor, Tensor], Optional[Tensor]]:
        raise NotImplementedError
