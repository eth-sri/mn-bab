from __future__ import annotations

from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from src.abstract_layers.abstract_container_module import AbstractContainerModule
from src.abstract_layers.abstract_module import AbstractModule
from src.abstract_layers.abstract_relu import ReLU
from src.exceptions.invalid_bounds import InvalidBoundsError
from src.mn_bab_shape import MN_BaB_Shape
from src.utilities.leaky_gradient_maximum_function import LeakyGradientMaximumFunction
from src.utilities.leaky_gradient_minimum_function import LeakyGradientMinimumFunction


class Sequential(AbstractContainerModule):
    def __init__(
        self,
        layers: Sequence[AbstractModule],
    ) -> None:
        super(Sequential, self).__init__()
        self.layers = nn.ModuleList(layers)
        *__, last_layer = layers
        self.output_dim = last_layer.output_dim
        self.set_dependence_set_applicability()

    @classmethod
    def from_concrete_module(
        cls,
        module: nn.Sequential,
        input_dim: Tuple[int, ...],
        **kwargs: Any,
    ) -> Sequential:
        assert "concrete_to_abstract" in kwargs
        concrete_to_abstract_mapping = kwargs["concrete_to_abstract"]
        abstract_layers: List[AbstractModule] = []
        for i, layer in enumerate(module.children()):
            if i == 0:
                current_layer_input_dim = input_dim
            else:
                current_layer_input_dim = abstract_layers[-1].output_dim

            abstract_type = concrete_to_abstract_mapping(type(layer))
            abstract_layers.append(
                abstract_type.from_concrete_module(
                    layer,
                    current_layer_input_dim,
                    **kwargs,
                )
            )
        return cls(abstract_layers)

    def forward(self, input: Tensor) -> Tensor:
        for layer in self.layers:
            input = layer(input)

        return input

    def reset_input_bounds(self) -> None:
        super(Sequential, self).reset_input_bounds()
        for layer in self.layers:
            layer.reset_input_bounds()

    def set_intermediate_input_bounds(
        self, intermediate_bounds: OrderedDict[int, Tuple[Tensor, Tensor]]
    ) -> None:
        for layer in self.layers:
            if id(layer) in intermediate_bounds:
                layer.update_input_bounds(
                    intermediate_bounds[id(layer)],
                )
            elif isinstance(layer, AbstractContainerModule):
                layer.set_intermediate_input_bounds(intermediate_bounds)

    def propagate_interval(
        self, interval: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tensor]:
        for layer in self.layers:
            layer.update_input_bounds(interval)
            interval = layer.propagate_interval(interval)
        return interval

    def set_dependence_set_applicability(self, applicable: bool = True) -> None:
        for layer in self.layers:
            if isinstance(layer, AbstractContainerModule):
                layer.set_dependence_set_applicability(applicable)
            else:
                layer.dependence_set_applicable = (
                    applicable and not layer.dependence_set_block
                )
            applicable = layer.dependence_set_applicable
        self.dependence_set_applicable = applicable

    def _get_mn_bab_shape_after_layer(
        self,
        from_layer_index: int,
        abstract_shape: MN_BaB_Shape,
        input_lb: Tensor,
        input_ub: Tensor,
        layer_ids_for_which_to_compute_prima_constraints: Optional[List[int]] = None,
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
    ) -> Tuple[Optional[MN_BaB_Shape], Optional[Tuple[Tensor, Tensor]]]:
        if layer_ids_for_which_to_compute_prima_constraints is None:
            layer_ids_for_which_to_compute_prima_constraints = []

        if from_layer_index == len(self.layers) - 1:
            starting_layer_id = id(self)
        else:
            starting_layer_id = id(self.layers[from_layer_index + 1])
        abstract_shape.set_optimizable_parameters(starting_layer_id)

        (
            abstract_shape,
            (best_lbs, best_ubs),
            unstable_queries,
        ) = self.backsubstitute_shape(
            abstract_shape,
            input_lb,
            input_ub,
            starting_layer_id,
            from_layer_index,
            layer_ids_for_which_to_compute_prima_constraints,
            best_intermediate_bounds_so_far,
            propagate_preceeding_callback,
            preceeding_layers,
            use_dependence_sets,
            use_early_termination,
            use_early_termination_for_current_query,
        )

        if propagate_preceeding_callback is not None:
            (
                abstract_shape,
                (lbs, ubs),
                unstable_queries,
            ) = propagate_preceeding_callback(
                abstract_shape,
                unstable_queries,
                use_early_termination,
                use_early_termination_for_current_query,
            )
            best_lbs = torch.maximum(best_lbs, lbs)
            best_ubs = torch.minimum(best_ubs, ubs)

        if use_early_termination_for_current_query:
            assert unstable_queries is not None
            if not unstable_queries.any():
                return None, (best_lbs, best_ubs)
            lbs, ubs = abstract_shape.concretize(input_lb, input_ub)
            if unstable_queries is not None:
                best_lbs[:, unstable_queries] = torch.maximum(
                    best_lbs[:, unstable_queries], lbs
                )
                best_ubs[:, unstable_queries] = torch.minimum(
                    best_ubs[:, unstable_queries], ubs
                )
            else:
                best_lbs = torch.maximum(best_lbs, lbs)
                best_ubs = torch.minimum(best_ubs, ubs)
            return None, (best_lbs, best_ubs)
        else:
            return abstract_shape, None

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
        if unstable_queries is None:
            number_of_queries = (
                abstract_shape.lb_coef.sets.shape[1]
                if abstract_shape.uses_dependence_sets()
                else abstract_shape.lb_coef.shape[1]  # type: ignore
            )
        else:
            number_of_queries = unstable_queries.shape[0]

        if use_early_termination_for_current_query:
            if unstable_queries is None:
                unstable_queries = torch.ones(
                    number_of_queries,
                    device=abstract_shape.device,
                    dtype=torch.bool,
                )
        abstract_shape.assert_matches_unstable_queries_mask(unstable_queries)

        best_lbs = torch.empty(
            (abstract_shape.batch_size, number_of_queries),
            device=abstract_shape.device,
        ).fill_(-float("inf"))
        best_ubs = torch.empty(
            (abstract_shape.batch_size, number_of_queries),
            device=abstract_shape.device,
        ).fill_(float("inf"))

        from_layer_index = (
            len(self.layers) - 1 if from_layer_index is None else from_layer_index
        )
        for i, layer in reversed(list(enumerate(self.layers[: from_layer_index + 1]))):
            if isinstance(layer, ReLU):
                intermediate_bounds_callback = None
                if id(layer) in layer_ids_for_which_to_compute_prima_constraints:
                    intermediate_bounds_callback = (
                        self._get_intermediate_bounds_callback(
                            i,
                            abstract_shape,
                            input_lb,
                            input_ub,
                            best_intermediate_bounds_so_far,
                            propagate_preceeding_callback,
                            use_dependence_sets,
                        )
                    )
                if layer.input_bounds is None:
                    try:
                        self._set_intermediate_bounds(
                            i,
                            abstract_shape.carried_over_optimizable_parameters,
                            abstract_shape.prima_coefficients,
                            abstract_shape.prima_hyperparameters,
                            abstract_shape.split_constraints,
                            abstract_shape.invalid_bounds_mask_in_batch,
                            abstract_shape.batch_size,
                            input_lb,
                            input_ub,
                            layer_ids_for_which_to_compute_prima_constraints,
                            abstract_shape.device,
                            use_dependence_sets,
                            use_early_termination,
                            best_intermediate_bounds_so_far,
                            propagate_preceeding_callback,
                            preceeding_layers,
                        )
                    except InvalidBoundsError as e:
                        abstract_shape.update_invalid_bounds_mask_in_batch(
                            e.invalid_bounds_mask_in_batch
                        )
                abstract_shape = layer.backsubstitute(
                    abstract_shape, intermediate_bounds_callback
                )

                if layer.input_bounds and use_early_termination_for_current_query:
                    lbs, ubs = abstract_shape.concretize(*layer.input_bounds)
                    best_lbs[:, unstable_queries] = torch.maximum(
                        best_lbs[:, unstable_queries], lbs
                    )
                    best_ubs[:, unstable_queries] = torch.minimum(
                        best_ubs[:, unstable_queries], ubs
                    )
                    current_unstable_queries = (lbs * ubs < 0).any(axis=0)
                    if not current_unstable_queries.all():
                        new_unstable_queries = unstable_queries.clone()
                        new_unstable_queries[unstable_queries] = (
                            current_unstable_queries
                            & unstable_queries[unstable_queries]
                        )
                        abstract_shape.filter_out_stable_queries(
                            current_unstable_queries
                        )
                        unstable_queries = new_unstable_queries
                        abstract_shape.assert_matches_unstable_queries_mask(
                            unstable_queries
                        )
                    elif not unstable_queries.any():
                        return abstract_shape, (best_lbs, best_ubs), unstable_queries

            elif isinstance(layer, AbstractContainerModule):
                if i == 0:
                    _propagate_preceeding_callback = propagate_preceeding_callback
                    _preceeding_layers = preceeding_layers
                else:

                    def _propagate_preceeding_callback(
                        abstract_shape_int: MN_BaB_Shape,
                        unstable_queries_int: Optional[Tensor],
                        use_early_termination: bool,
                        use_early_termination_for_current_query: bool,
                    ) -> Tuple[MN_BaB_Shape, Tuple[Tensor, Tensor], Optional[Tensor]]:
                        (
                            abstract_shape_int,
                            (lbs, ubs),
                            unstable_queries_int,
                        ) = self.backsubstitute_shape(
                            abstract_shape_int,
                            input_lb,
                            input_ub,
                            starting_layer_id,
                            i - 1,
                            layer_ids_for_which_to_compute_prima_constraints,
                            best_intermediate_bounds_so_far=best_intermediate_bounds_so_far,
                            propagate_preceeding_callback=propagate_preceeding_callback,
                            preceeding_layers=preceeding_layers,
                            use_dependence_sets=use_dependence_sets,
                            use_early_termination=use_early_termination,
                            use_early_termination_for_current_query=use_early_termination_for_current_query,
                            full_back_prop=True,
                            unstable_queries=unstable_queries_int,
                        )
                        return abstract_shape_int, (lbs, ubs), unstable_queries_int

                    _preceeding_layers = [preceeding_layers, self.layers[:i]]

                (
                    abstract_shape,
                    (lbs, ubs),
                    unstable_queries,
                ) = layer.backsubstitute_shape(
                    abstract_shape,
                    input_lb,
                    input_ub,
                    starting_layer_id,
                    None,
                    layer_ids_for_which_to_compute_prima_constraints,
                    best_intermediate_bounds_so_far,
                    _propagate_preceeding_callback,
                    _preceeding_layers,
                    use_dependence_sets,
                    use_early_termination,
                    use_early_termination_for_current_query,
                    unstable_queries=unstable_queries,
                )
                best_lbs = torch.maximum(best_lbs, lbs)
                best_ubs = torch.minimum(best_ubs, ubs)
            else:
                abstract_shape = layer.backsubstitute(abstract_shape)

        if propagate_preceeding_callback is not None and full_back_prop:
            (
                abstract_shape,
                (lbs, ubs),
                unstable_queries,
            ) = propagate_preceeding_callback(
                abstract_shape,
                unstable_queries,
                use_early_termination,
                use_early_termination_for_current_query,
            )
            best_lbs = torch.maximum(best_lbs, lbs)
            best_ubs = torch.minimum(best_ubs, ubs)
        return abstract_shape, (best_lbs, best_ubs), unstable_queries

    def _set_intermediate_bounds(
        self,
        current_layer_index: int,
        carried_over_optimizable_parameters: Optional[
            Dict[int, Dict[str, Dict[int, Tensor]]]
        ],
        prima_coefficients: Optional[Dict[int, Tuple[Tensor, Tensor, Tensor]]],
        prima_hyperparameters: Optional[Dict[str, float]],
        split_constraints: Optional[Dict[int, Tensor]],
        invalid_bounds_mask_in_batch: Optional[Sequence[bool]],
        batch_size: int,
        input_lb: Tensor,
        input_ub: Tensor,
        layer_ids_for_which_to_compute_prima_constraints: List[int],
        device: torch.device,
        use_dependence_sets: bool,
        use_early_termination: bool,
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
    ) -> None:
        if current_layer_index == 0:
            assert preceeding_layers is not None
            prev_layer = preceeding_layers[-1][-1]
        else:
            prev_layer = self.layers[current_layer_index - 1]
        # TODO for the use of intermediate bounds get the actual prev layer in case of Sequential or BB
        current_layer = self.layers[current_layer_index]

        use_dependence_sets_for_current_bounds = (
            use_dependence_sets
            and self.layers[current_layer_index].dependence_set_applicable
        )

        if (
            split_constraints is not None
            and best_intermediate_bounds_so_far is not None
            and id(current_layer) in best_intermediate_bounds_so_far
        ):
            unstable_nodes_in_current_layer = (
                split_constraints[id(current_layer)] == 0
            ).any(axis=0)
            intermediate_bounds_to_recompute = unstable_nodes_in_current_layer.flatten()
        else:
            intermediate_bounds_to_recompute = torch.ones(
                current_layer.output_dim,
                device=device,
                dtype=torch.bool,
            ).flatten()

        if not intermediate_bounds_to_recompute.any():
            assert (
                best_intermediate_bounds_so_far is not None
                and id(current_layer) in best_intermediate_bounds_so_far
            )

            current_layer.update_input_bounds(
                best_intermediate_bounds_so_far[id(current_layer)]
            )
            return

        abstract_shape = MN_BaB_Shape.construct_to_bound_all_outputs(
            device,
            prev_layer.output_dim,
            batch_size,
            carried_over_optimizable_parameters,
            prima_coefficients,
            prima_hyperparameters,
            split_constraints,
            invalid_bounds_mask_in_batch,
            use_dependence_sets_for_current_bounds,
        )
        if not intermediate_bounds_to_recompute.all():
            abstract_shape.filter_out_stable_queries(intermediate_bounds_to_recompute)
        propagated_shape, layer_bounds = self._get_mn_bab_shape_after_layer(
            current_layer_index - 1,
            abstract_shape,
            input_lb,
            input_ub,
            layer_ids_for_which_to_compute_prima_constraints,
            best_intermediate_bounds_so_far=best_intermediate_bounds_so_far,
            propagate_preceeding_callback=propagate_preceeding_callback,
            preceeding_layers=preceeding_layers,
            use_dependence_sets=use_dependence_sets,
            use_early_termination=use_early_termination,
            use_early_termination_for_current_query=use_early_termination,
        )
        if propagated_shape is not None:
            assert layer_bounds is None
            recomputed_layer_lb, recomputed_layer_ub = propagated_shape.concretize(
                input_lb, input_ub
            )
        else:
            assert layer_bounds is not None
            recomputed_layer_lb, recomputed_layer_ub = layer_bounds

        if not intermediate_bounds_to_recompute.all():
            assert (
                best_intermediate_bounds_so_far is not None
                and id(current_layer) in best_intermediate_bounds_so_far
            )
            layer_lb = (
                best_intermediate_bounds_so_far[id(current_layer)][0]
                .flatten(start_dim=1)
                .clone()
            )
            layer_ub = (
                best_intermediate_bounds_so_far[id(current_layer)][1]
                .flatten(start_dim=1)
                .clone()
            )

            layer_lb[:, intermediate_bounds_to_recompute] = recomputed_layer_lb
            layer_ub[:, intermediate_bounds_to_recompute] = recomputed_layer_ub
        else:
            layer_lb, layer_ub = recomputed_layer_lb, recomputed_layer_ub

        if (
            best_intermediate_bounds_so_far is not None
            and id(current_layer) in best_intermediate_bounds_so_far
        ):
            best_lb, best_ub = best_intermediate_bounds_so_far[id(current_layer)]
            layer_lb = LeakyGradientMaximumFunction.apply(
                layer_lb, best_lb.flatten(start_dim=1)
            )
            layer_ub = LeakyGradientMinimumFunction.apply(
                layer_ub, best_ub.flatten(start_dim=1)
            )

        current_layer.update_input_bounds((layer_lb, layer_ub))

    def _get_intermediate_bounds_callback(
        self,
        layer_index: int,
        abstract_shape: MN_BaB_Shape,
        input_lb: Tensor,
        input_ub: Tensor,
        best_intermediate_bounds_so_far: Optional[
            OrderedDict[int, Tuple[Tensor, Tensor]]
        ] = None,
        propagate_preceeding_callback: Optional[
            Callable[
                [MN_BaB_Shape, Optional[Tensor], bool, bool],
                Tuple[MN_BaB_Shape, Tuple[Tensor, Tensor], Optional[Tensor]],
            ]
        ] = None,
        use_dependence_sets: bool = False,
    ) -> Callable[[Tensor], Tuple[Tensor, Tensor]]:
        assert layer_index >= 1
        layer_input_shape = self.layers[layer_index - 1].output_dim

        @torch.no_grad()
        def compute_intermediate_bounds(
            query_coef: Tensor,
        ) -> Tuple[Tensor, Tensor]:
            query_coef = query_coef.view(*(query_coef.shape[:2] + layer_input_shape))
            query_shape = MN_BaB_Shape.construct_with_same_optimization_state_as(
                abstract_shape, lb_coef=query_coef, ub_coef=query_coef
            )

            propagated_shape, __ = self._get_mn_bab_shape_after_layer(
                layer_index - 1,
                query_shape,
                input_lb,
                input_ub,
                best_intermediate_bounds_so_far=best_intermediate_bounds_so_far,
                propagate_preceeding_callback=propagate_preceeding_callback,
                use_dependence_sets=use_dependence_sets,
                use_early_termination=False,
            )
            # assert query_shape.carried_over_optimizable_parameters == abstract_shape.carried_over_optimizable_parameters
            assert propagated_shape is not None
            return propagated_shape.concretize(input_lb, input_ub)

        return compute_intermediate_bounds

    def get_default_split_constraints(self) -> Dict[int, Tensor]:
        split_contstraints = {}
        for layer in self.layers:
            if isinstance(layer, ReLU):
                split_contstraints[id(layer)] = torch.zeros(
                    1, *layer.output_dim, dtype=torch.int8
                )
            elif isinstance(layer, AbstractContainerModule):
                split_contstraints.update(layer.get_default_split_constraints())

        return split_contstraints

    def get_current_intermediate_bounds(
        self,
    ) -> OrderedDict[int, Tuple[Tensor, Tensor]]:
        intermediate_bounds = OrderedDict()
        for layer in self.layers:
            if layer.input_bounds is not None:
                intermediate_bounds[id(layer)] = (
                    layer.input_bounds[0].detach(),
                    layer.input_bounds[1].detach(),
                )
            elif isinstance(layer, AbstractContainerModule):
                intermediate_bounds.update(layer.get_current_intermediate_bounds())
        return intermediate_bounds

    def get_babsr_bias(self, from_layer_index: Optional[int] = None) -> Tensor:
        if from_layer_index is None:
            from_layer_index = len(self.layers) - 1
        for i, layer in reversed(list(enumerate(self.layers[: from_layer_index + 1]))):
            if isinstance(layer, Sequential):
                return layer.get_babsr_bias()
            elif hasattr(layer, "bias"):
                return layer.bias

    def get_activation_layer_ids(
        self, act_layer_ids: Optional[List[int]] = None
    ) -> List[int]:
        if act_layer_ids is None:
            act_layer_ids = []
        for layer in self.layers:
            act_layer_ids += layer.get_activation_layer_ids()
        return act_layer_ids
