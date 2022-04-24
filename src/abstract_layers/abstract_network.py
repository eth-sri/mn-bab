from __future__ import annotations

from collections import OrderedDict
from typing import Any, List, Optional, Sequence, Tuple

from torch import Tensor, nn

from src.abstract_layers.abstract_module import AbstractModule
from src.abstract_layers.abstract_sequential import Sequential
from src.mn_bab_shape import MN_BaB_Shape
from src.utilities.abstract_module_mapper import AbstractModuleMapper


class AbstractNetwork(Sequential):
    def __init__(
        self,
        layers: Sequence[AbstractModule],
    ) -> None:
        super(AbstractNetwork, self).__init__(layers)

    @classmethod
    def from_concrete_module(
        cls, module: nn.Sequential, input_dim: Tuple[int, ...], **kwargs: Any
    ) -> AbstractNetwork:
        layers = Sequential.from_concrete_module(
            module,
            input_dim,
            concrete_to_abstract=AbstractModuleMapper.map_to_abstract_type,
        ).layers
        return cls(layers)

    def get_mn_bab_shape(
        self,
        input_lb: Tensor,
        input_ub: Tensor,
        abstract_shape: MN_BaB_Shape,
        intermediate_layers_to_be_kept_fixed: Optional[Sequence[int]] = None,
        best_intermediate_bounds_so_far: Optional[
            OrderedDict[int, Tuple[Tensor, Tensor]]
        ] = None,
        layer_ids_for_which_to_compute_prima_constraints: Optional[List[int]] = None,
        use_dependence_sets: bool = False,
        use_early_termination: bool = False,
    ) -> MN_BaB_Shape:
        self.reset_input_bounds()
        if intermediate_layers_to_be_kept_fixed is not None:
            assert best_intermediate_bounds_so_far is not None
            fixed_intermediate_bounds = OrderedDict(
                (layer_id, bounds)
                for layer_id, bounds in best_intermediate_bounds_so_far.items()
                if layer_id in intermediate_layers_to_be_kept_fixed
            )
            self.set_intermediate_input_bounds(fixed_intermediate_bounds)
        if layer_ids_for_which_to_compute_prima_constraints is None:
            layer_ids_for_which_to_compute_prima_constraints = []
        shape, __ = self._get_mn_bab_shape_after_layer(
            len(self.layers) - 1,
            abstract_shape,
            input_lb,
            input_ub,
            layer_ids_for_which_to_compute_prima_constraints,
            best_intermediate_bounds_so_far=best_intermediate_bounds_so_far,
            use_dependence_sets=use_dependence_sets,
            use_early_termination=use_early_termination,
            use_early_termination_for_current_query=False,
        )
        assert shape is not None
        return shape

    def set_layer_bounds_via_interval_propagation(
        self, input_lb: Tensor, input_ub: Tensor
    ) -> None:
        if len(input_lb.shape) in [2, 4]:
            shape_with_batch_dimension = input_lb.shape
        elif len(input_lb.shape) in [1, 3]:
            shape_with_batch_dimension = 1, *(input_lb.shape)
        else:
            raise RuntimeError(
                "Unexpected number of dimensions for interval propagation."
            )
        interval = (
            input_lb.expand(shape_with_batch_dimension),
            input_ub.expand(shape_with_batch_dimension),
        )
        for layer in self.layers:
            layer.update_input_bounds(interval)
            interval = layer.propagate_interval(interval)
