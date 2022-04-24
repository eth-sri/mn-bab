from __future__ import annotations

from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from src.abstract_layers.abstract_container_module import AbstractContainerModule
from src.abstract_layers.abstract_sequential import Sequential
from src.concrete_layers import basic_block as concrete_basic_block
from src.concrete_layers.basic_block import BasicBlock as concreteBasicBlock
from src.mn_bab_shape import MN_BaB_Shape


class BasicBlock(concreteBasicBlock, AbstractContainerModule):
    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int,
        bn: bool,
        kernel: int,
        input_dim: Tuple[int, ...],
        **kwargs: Any,
    ) -> None:
        super(BasicBlock, self).__init__(in_planes, planes, stride, bn, kernel)
        self.path_a = Sequential.from_concrete_module(self.path_a, input_dim, **kwargs)
        self.path_b = Sequential.from_concrete_module(self.path_b, input_dim, **kwargs)
        self.output_dim = self.path_b.layers[-1].output_dim
        self.bias = self.get_babsr_bias()

    @classmethod
    def from_concrete_module(
        cls,
        module: concrete_basic_block.BasicBlock,
        input_dim: Tuple[int, ...],
        **kwargs: Any,
    ) -> BasicBlock:
        abstract_layer = cls(
            module.in_planes,
            module.planes,
            module.stride,
            module.bn,
            module.kernel,
            input_dim,
            **kwargs,
        )
        abstract_layer.path_a = Sequential.from_concrete_module(
            module.path_a, input_dim, **kwargs
        )
        abstract_layer.path_b = Sequential.from_concrete_module(
            module.path_b, input_dim, **kwargs
        )
        abstract_layer.bias = abstract_layer.get_babsr_bias()
        return abstract_layer

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
        in_lb_bias = abstract_shape.lb_bias.clone()
        in_ub_bias = abstract_shape.ub_bias.clone()
        in_lb_coef = (
            abstract_shape.lb_coef.clone()
        )  # Also works properly for dependence set
        in_ub_coef = abstract_shape.ub_coef.clone()

        (
            a_shape_a,
            (lbs_a, ubs_a),
            unstable_queries_new,
        ) = self.path_a.backsubstitute_shape(
            abstract_shape,
            input_lb,
            input_ub,
            starting_layer_id,
            None,
            layer_ids_for_which_to_compute_prima_constraints,
            best_intermediate_bounds_so_far,
            propagate_preceeding_callback,
            preceeding_layers,
            use_dependence_sets=use_dependence_sets,
            use_early_termination=use_early_termination,
            use_early_termination_for_current_query=False,
            unstable_queries=unstable_queries,
        )
        assert (
            unstable_queries_new is None
            or (unstable_queries_new == unstable_queries).all()
        )

        a_lb_bias = a_shape_a.lb_bias.clone()
        a_ub_bias = a_shape_a.ub_bias.clone()
        a_lb_coef = a_shape_a.lb_coef.clone()
        a_ub_coef = a_shape_a.ub_coef.clone()

        abstract_shape.update_bounds(in_lb_coef, in_ub_coef, in_lb_bias, in_ub_bias)
        a_shape_b, __, unstable_queries_new = self.path_b.backsubstitute_shape(
            abstract_shape,
            input_lb,
            input_ub,
            starting_layer_id,
            None,
            layer_ids_for_which_to_compute_prima_constraints,
            best_intermediate_bounds_so_far,
            propagate_preceeding_callback,
            preceeding_layers,
            use_dependence_sets=use_dependence_sets,
            use_early_termination=use_early_termination,
            use_early_termination_for_current_query=False,
            unstable_queries=unstable_queries,
        )
        assert (
            unstable_queries_new is None
            or (unstable_queries_new == unstable_queries).all()
        )

        new_lb_bias = (
            a_lb_bias + a_shape_b.lb_bias - in_lb_bias
        )  # Both the shape in a and in b  contain the initial bias terms, so one has to be subtracted
        new_ub_bias = a_ub_bias + a_shape_b.ub_bias - in_ub_bias
        new_lb_coef = a_lb_coef + a_shape_b.lb_coef
        new_ub_coef = a_ub_coef + a_shape_b.ub_coef

        abstract_shape.update_bounds(new_lb_coef, new_ub_coef, new_lb_bias, new_ub_bias)
        return (
            abstract_shape,
            (-np.inf * torch.ones_like(lbs_a), np.inf * torch.ones_like(lbs_a)),
            unstable_queries,
        )

    def get_babsr_bias(self) -> Tensor:
        bias_a = self.path_a.get_babsr_bias()
        bias_b = self.path_b.get_babsr_bias()
        if bias_a is None:
            if bias_b is None:
                return 0
            else:
                return bias_b
        elif bias_b is None:
            return bias_a
        else:
            return bias_a + bias_b

    def reset_input_bounds(self) -> None:
        super(BasicBlock, self).reset_input_bounds()
        self.path_a.reset_input_bounds()
        self.path_b.reset_input_bounds()

    def propagate_interval(
        self, interval: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tensor]:
        interval_a = self.path_a.propagate_interval(interval)
        interval_b = self.path_b.propagate_interval(interval)

        return interval_a[0] + interval_b[0], interval_a[1] + interval_b[1]

    def set_dependence_set_applicability(self, applicable: bool = True) -> None:
        self.path_a.set_dependence_set_applicability(applicable)
        self.path_b.set_dependence_set_applicability(applicable)
        self.dependence_set_applicable = (
            self.path_a.layers[-1].dependence_set_applicable
            and self.path_b.layers[-1].dependence_set_applicable
        )

    def get_default_split_constraints(self) -> Dict[int, Tensor]:
        split_contstraints = {}
        split_contstraints.update(self.path_a.get_default_split_constraints())
        split_contstraints.update(self.path_b.get_default_split_constraints())
        return split_contstraints

    def get_current_intermediate_bounds(
        self,
    ) -> OrderedDict[int, Tuple[Tensor, Tensor]]:
        intermediate_bounds = OrderedDict()
        intermediate_bounds.update(self.path_a.get_current_intermediate_bounds())
        intermediate_bounds.update(self.path_b.get_current_intermediate_bounds())
        return intermediate_bounds

    def set_intermediate_input_bounds(
        self, intermediate_bounds: OrderedDict[int, Tuple[Tensor, Tensor]]
    ) -> None:
        self.path_a.set_intermediate_input_bounds(intermediate_bounds)
        self.path_b.set_intermediate_input_bounds(intermediate_bounds)

    def get_activation_layer_ids(
        self, act_layer_ids: Optional[List[int]] = None
    ) -> List[int]:
        if act_layer_ids is None:
            act_layer_ids = []
        act_layer_ids += self.path_a.get_activation_layer_ids()
        act_layer_ids += self.path_b.get_activation_layer_ids()
        return act_layer_ids
