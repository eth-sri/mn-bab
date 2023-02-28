from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from src.abstract_domains.ai_util import AbstractElement
from src.abstract_layers.abstract_container_module import (
    AbstractContainerModule,
    ActivationLayer,
    ActivationLayers,
)
from src.abstract_layers.abstract_identity import Identity
from src.abstract_layers.abstract_max_pool2d import MaxPool2d
from src.abstract_layers.abstract_module import AbstractModule
from src.abstract_layers.abstract_relu import ReLU
from src.abstract_layers.abstract_sigmoid import Sigmoid
from src.abstract_layers.abstract_tanh import Tanh
from src.exceptions.invalid_bounds import InvalidBoundsError
from src.mn_bab_shape import AffineForm, MN_BaB_Shape
from src.state.tags import (
    LayerTag,
    QueryTag,
    layer_from_query_tag,
    layer_tag,
    query_tag,
)

# from src.state.tags import query_tag_for_neuron
from src.utilities.config import BacksubstitutionConfig
from src.utilities.dependence_sets import DependenceSets

# from src.utilities.config import ParameterSharing
# from src.utilities.layer_types import is_layer_of_type
from src.utilities.leaky_gradient_maximum_function import LeakyGradientMaximumFunction
from src.utilities.leaky_gradient_minimum_function import LeakyGradientMinimumFunction
from src.utilities.queries import (
    QueryCoef,
    get_output_bound_initial_query_coef_iterator,
    num_queries,
)
from src.verification_subproblem import SubproblemState


class Sequential(AbstractContainerModule):
    def __init__(
        self,
        layers: Iterable[AbstractModule],
    ) -> None:
        super(Sequential, self).__init__()
        self.layers = nn.ModuleList(layers)
        # Build during backsubstitution
        # self.layer_id_to_layer: Dict[LayerTag, ActivationLayer] = {}
        *__, last_layer = layers
        self.output_dim = last_layer.output_dim
        self.set_dependence_set_applicability()

    @classmethod
    def from_concrete_module(  # type: ignore[override] # (checked at runtime)
        cls,
        module: nn.Sequential,
        input_dim: Tuple[int, ...],
        **kwargs: Any,
    ) -> Sequential:
        assert isinstance(module, nn.Sequential)
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
        if len(abstract_layers) == 0:
            abstract_layers.append(Identity(input_dim=input_dim))
        return cls(abstract_layers)

    def forward(self, input: Tensor) -> Tensor:
        for layer in self.layers:
            input = layer(input)

        return input

    def reset_input_bounds(self) -> None:
        super(Sequential, self).reset_input_bounds()
        for layer in self.layers:
            layer.reset_input_bounds()

    def reset_optim_input_bounds(self) -> None:
        super(Sequential, self).reset_input_bounds()
        for layer in self.layers:
            layer.reset_optim_input_bounds()

    def reset_output_bounds(self) -> None:
        super(Sequential, self).reset_output_bounds()
        for layer in self.layers:
            layer.reset_output_bounds()

    def set_intermediate_input_bounds(
        self, intermediate_bounds: OrderedDict[LayerTag, Tuple[Tensor, Tensor]]
    ) -> None:
        for layer in self.layers:
            if layer_tag(layer) in intermediate_bounds:
                layer.update_input_bounds(
                    intermediate_bounds[layer_tag(layer)],
                    check_feasibility=False,
                )
            if isinstance(layer, AbstractContainerModule):
                layer.set_intermediate_input_bounds(intermediate_bounds)

    def propagate_interval(
        self,
        interval: Tuple[Tensor, Tensor],
        use_existing_bounds: Optional[bool] = None,
        subproblem_state: Optional[SubproblemState] = None,
        activation_layer_only: bool = False,
        set_input: bool = True,
        set_output: bool = True,
    ) -> Tuple[Tensor, Tensor]:
        for layer in self.layers:

            if use_existing_bounds and layer.input_bounds is not None:
                lb = torch.max(interval[0], layer.input_bounds[0])
                ub = torch.min(interval[1], layer.input_bounds[1])
                interval = (lb, ub)

            if use_existing_bounds and layer.optim_input_bounds is not None:
                lb = torch.max(interval[0], layer.optim_input_bounds[0])
                ub = torch.min(interval[1], layer.optim_input_bounds[1])
                interval = (lb, ub)

            if set_input and (
                type(layer) in ActivationLayers or not activation_layer_only
            ):
                layer.update_input_bounds(interval, check_feasibility=False)
            interval = layer.propagate_interval(
                interval,
                use_existing_bounds,
                activation_layer_only=activation_layer_only,
                set_input=set_input,
                set_output=set_output,
            )
            assert (interval[0] <= interval[1]).all()
            if set_output and (
                type(layer) in ActivationLayers or not activation_layer_only
            ):
                layer.update_output_bounds(interval)
        return interval

    def propagate_abstract_element(
        self,
        abs_input: AbstractElement,
        use_existing_bounds: Optional[bool] = None,
        activation_layer_only: bool = False,
        set_input: bool = True,
        set_output: bool = True,
    ) -> AbstractElement:
        for layer in self.layers:
            if (
                set_input
                and type(layer) in ActivationLayers
                or not activation_layer_only
            ):
                layer.update_input_bounds(
                    abs_input.concretize(), check_feasibility=False
                )
            abs_input = layer.propagate_abstract_element(
                abs_input,
                use_existing_bounds,
                activation_layer_only,
                set_input,
                set_output,
            )
            if (
                set_output
                and type(layer) in ActivationLayers
                or not activation_layer_only
            ):
                layer.update_output_bounds(abs_input.concretize())
        return abs_input

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
        config: BacksubstitutionConfig,
        input_lb: Tensor,
        input_ub: Tensor,
        abstract_shape: MN_BaB_Shape,
        propagate_preceeding_callback: Optional[
            Callable[
                [BacksubstitutionConfig, MN_BaB_Shape, bool],
                Tuple[MN_BaB_Shape, Tuple[Tensor, Tensor]],
            ]
        ],
        preceeding_layers: Optional[List[Any]],
        use_early_termination_for_current_query: bool,  # = False,
        optimize_intermediate_bounds: bool,  # = False,  # Whether to run individual backward passes to tighten intermediate bounds individually
    ) -> Tuple[Optional[MN_BaB_Shape], Optional[Tuple[Tensor, Tensor]]]:

        if from_layer_index == len(self.layers) - 1:
            assert layer_from_query_tag(abstract_shape.query_id) == layer_tag(
                self
            )  # TODO: this seems a bit messy
        else:
            assert layer_from_query_tag(abstract_shape.query_id) == layer_tag(
                self.layers[from_layer_index + 1]
            )  # This kind of makes sense: this is the tag of the node for which you get input constraints

        (abstract_shape, (best_lbs, best_ubs)) = self.backsubstitute_shape(
            config=config,
            input_lb=input_lb,
            input_ub=input_ub,
            abstract_shape=abstract_shape,
            from_layer_index=from_layer_index,
            propagate_preceeding_callback=propagate_preceeding_callback,
            preceeding_layers=preceeding_layers,
            use_early_termination_for_current_query=use_early_termination_for_current_query,
            full_back_prop=True,
            optimize_intermediate_bounds=optimize_intermediate_bounds,
        )

        if use_early_termination_for_current_query:
            assert abstract_shape.unstable_queries is not None
            if not abstract_shape.unstable_queries.any():
                return None, (best_lbs, best_ubs)
            # TODO: move into MN_BaB_shape, this logic is duplicated
            curr_lbs, curr_ubs = abstract_shape.concretize(input_lb, input_ub)
            assert (
                curr_ubs is not None
            )  # No early termination when we have only lower bounds (atm)
            lbs, ubs = curr_lbs, curr_ubs
            if abstract_shape.unstable_queries is not None:  # TODO: this is always true
                best_lbs[:, abstract_shape.unstable_queries] = torch.maximum(
                    best_lbs[:, abstract_shape.unstable_queries], lbs
                )
                best_ubs[:, abstract_shape.unstable_queries] = torch.minimum(
                    best_ubs[:, abstract_shape.unstable_queries], ubs
                )
            else:
                best_lbs = torch.maximum(best_lbs, lbs)
                best_ubs = torch.minimum(best_ubs, ubs)
                # TODO @Robin
            return None, (best_lbs, best_ubs)
        else:
            return abstract_shape, None

    def backsubstitute_shape(  # noqa: C901
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

        number_of_queries = abstract_shape.num_queries
        assert (
            abstract_shape.unstable_queries is None
            or abstract_shape.unstable_queries.sum() == number_of_queries
        )

        if use_early_termination_for_current_query:
            if (
                abstract_shape.unstable_queries is None
            ):  # TODO: initialize this eagerly?
                abstract_shape.initialize_unstable_queries()
            assert abstract_shape.unstable_queries is not None
            assert abstract_shape.matches_filter_mask(
                abstract_shape.unstable_queries
            )  # TODO: comment out for performance?

        best_lbs = torch.empty(  # TODO: move into MN_BaB_Shape
            (
                abstract_shape.batch_size,
                number_of_queries
                if abstract_shape.unstable_queries is None
                else len(abstract_shape.unstable_queries),
            ),
            device=abstract_shape.device,
        ).fill_(-torch.inf)
        best_ubs = torch.empty(
            (
                abstract_shape.batch_size,
                number_of_queries
                if abstract_shape.unstable_queries is None
                else len(abstract_shape.unstable_queries),
            ),
            device=abstract_shape.device,
        ).fill_(torch.inf)

        from_layer_index = (
            len(self.layers) - 1 if from_layer_index is None else from_layer_index
        )
        for i, layer in reversed(list(enumerate(self.layers[: from_layer_index + 1]))):
            if isinstance(abstract_shape.lb.coef, DependenceSets):
                if np.prod(layer.output_dim[-2:]) <= np.prod(
                    abstract_shape.lb.coef.sets.shape[-2:]
                ):
                    abstract_shape.lb = AffineForm(
                        abstract_shape.lb.coef.to_tensor(layer.output_dim),
                        abstract_shape.lb.bias,
                    )
                    if abstract_shape.ub is not None:
                        assert isinstance(abstract_shape.ub.coef, DependenceSets)
                        abstract_shape.ub = AffineForm(
                            abstract_shape.ub.coef.to_tensor(layer.output_dim),
                            abstract_shape.ub.bias,
                        )

            if (
                isinstance(layer, ReLU)
                or isinstance(layer, Sigmoid)
                or isinstance(layer, Tanh)
                or isinstance(layer, MaxPool2d)
            ):
                # if not layer_tag(layer) in self.layer_id_to_layer:
                #   self.layer_id_to_layer[layer_tag(layer)] = layer
                intermediate_bounds_callback = None
                if (
                    layer_tag(layer)
                    in config.layer_ids_for_which_to_compute_prima_constraints
                ):
                    intermediate_bounds_callback = (
                        self._get_intermediate_bounds_callback(
                            layer_index=i,
                            config=config,
                            subproblem_state=abstract_shape.subproblem_state,
                            device=abstract_shape.device,
                            input_lb=input_lb,
                            input_ub=input_ub,
                            propagate_preceeding_callback=propagate_preceeding_callback,
                            preceeding_layers=preceeding_layers,
                        )
                    )
                if layer.input_bounds is None:
                    # print(f"Setting intermediate bounds for layer {i}")
                    try:
                        self._set_intermediate_bounds(
                            current_layer_index=i,
                            config=config,
                            input_lb=input_lb,
                            input_ub=input_ub,
                            batch_size=abstract_shape.batch_size,
                            subproblem_state=abstract_shape.subproblem_state,
                            device=abstract_shape.device,
                            propagate_preceeding_callback=propagate_preceeding_callback,
                            preceeding_layers=preceeding_layers,
                            optimize_intermediate_bounds=optimize_intermediate_bounds,
                        )
                    except InvalidBoundsError as e:
                        abstract_shape.update_is_infeasible(
                            e.invalid_bounds_mask_in_batch
                        )
                    # print(f"Finished intermediate bounds for layer {i}")
                abstract_shape = layer.backsubstitute(
                    config,
                    abstract_shape,
                    intermediate_bounds_callback,
                    self.get_prev_layer(i, preceeding_layers),
                )

                if layer.input_bounds and use_early_termination_for_current_query:
                    # TODO: move into MN_BaB_Shape, this logic is duplicated
                    assert abstract_shape.unstable_queries is not None
                    curr_lbs, curr_ubs = abstract_shape.concretize(*layer.input_bounds)
                    assert (
                        curr_ubs is not None
                    )  # No early termination when we have only lower bounds (atm)
                    lbs, ubs = curr_lbs, curr_ubs
                    best_lbs[:, abstract_shape.unstable_queries] = torch.maximum(
                        best_lbs[:, abstract_shape.unstable_queries], lbs
                    )
                    best_ubs[:, abstract_shape.unstable_queries] = torch.minimum(
                        best_ubs[:, abstract_shape.unstable_queries], ubs
                    )
                    if isinstance(layer, ReLU):
                        current_unstable_queries = (lbs * ubs < 0).any(
                            dim=0
                        )  # was: axis=0
                        # print(f"Before2: {unstable_queries.shape}")
                        abstract_shape.update_unstable_queries(current_unstable_queries)
                        if not abstract_shape.unstable_queries.any():
                            # if no current queries are unstable, we can return early
                            return (
                                abstract_shape,
                                (best_lbs, best_ubs),
                            )

            elif isinstance(layer, AbstractContainerModule):
                if i == 0:
                    _propagate_preceeding_callback = propagate_preceeding_callback
                    _preceeding_layers = preceeding_layers
                else:

                    def _propagate_preceeding_callback(
                        config: BacksubstitutionConfig,
                        abstract_shape_int: MN_BaB_Shape,
                        use_early_termination_for_current_query: bool,
                    ) -> Tuple[MN_BaB_Shape, Tuple[Tensor, Tensor]]:
                        (abstract_shape_int, (lbs, ubs),) = self.backsubstitute_shape(
                            config=config,
                            input_lb=input_lb,
                            input_ub=input_ub,
                            abstract_shape=abstract_shape_int,
                            from_layer_index=i - 1,
                            propagate_preceeding_callback=propagate_preceeding_callback,
                            preceeding_layers=preceeding_layers,
                            use_early_termination_for_current_query=use_early_termination_for_current_query,
                            full_back_prop=True,
                            optimize_intermediate_bounds=optimize_intermediate_bounds,
                        )
                        return abstract_shape_int, (lbs, ubs)

                    _preceeding_layers = [preceeding_layers, self.layers[:i]]

                (abstract_shape, (lbs, ubs),) = layer.backsubstitute_shape(
                    config=config,
                    input_lb=input_lb,
                    input_ub=input_ub,
                    abstract_shape=abstract_shape,
                    from_layer_index=None,
                    propagate_preceeding_callback=_propagate_preceeding_callback,
                    preceeding_layers=_preceeding_layers,
                    use_early_termination_for_current_query=use_early_termination_for_current_query,
                    full_back_prop=False,
                    optimize_intermediate_bounds=optimize_intermediate_bounds,
                )
                if (  # TODO: move into MN_BaB_Shape
                    lbs.shape != best_lbs.shape
                    and abstract_shape.unstable_queries is not None
                    and abstract_shape.unstable_queries.sum() < best_lbs.numel()
                ):
                    best_lbs[:, abstract_shape.unstable_queries] = torch.maximum(
                        best_lbs[:, abstract_shape.unstable_queries], lbs
                    )
                    best_ubs[:, abstract_shape.unstable_queries] = torch.minimum(
                        best_ubs[:, abstract_shape.unstable_queries], ubs
                    )
                else:
                    best_lbs = torch.maximum(best_lbs, lbs)
                    best_ubs = torch.minimum(best_ubs, ubs)
            else:
                # print(f"Pre layer {i} - {layer}: Shape: {abstract_shape.lb.coef.shape}")
                # print(f"affine form dtype: {abstract_shape.lb.coef.dtype}")
                abstract_shape = layer.backsubstitute(config, abstract_shape)
                # print(f"Post layer {i} - {layer}: Shape: {abstract_shape.lb.coef.shape}")
                # print(f"affine form dtype: {abstract_shape.lb.coef.dtype}")

        if propagate_preceeding_callback is not None and full_back_prop:
            (abstract_shape, (lbs, ubs),) = propagate_preceeding_callback(
                config,
                abstract_shape,
                use_early_termination_for_current_query,
            )
            best_lbs = torch.maximum(best_lbs, lbs)
            best_ubs = torch.minimum(best_ubs, ubs)
        return abstract_shape, (best_lbs, best_ubs)

    def get_prev_layer(
        self,
        current_layer_index: int,
        preceeding_layers: Optional[List[Any]],
    ) -> AbstractModule:
        if current_layer_index == 0:
            # Reached the beginning of a sequential block. Preceeding layer is input to this block.
            assert preceeding_layers is not None
            if preceeding_layers is None:
                return None
            else:
                prev_layer = preceeding_layers[-1][-1]
        else:
            prev_layer = self.layers[current_layer_index - 1]
        assert isinstance(prev_layer, AbstractModule)
        # TODO for the use of intermediate bounds get the actual prev layer in case of Sequential or BB
        return prev_layer

    def _set_intermediate_bounds(  # noqa: C901 # TODO: simplify
        self,
        current_layer_index: int,
        config: BacksubstitutionConfig,
        input_lb: Tensor,
        input_ub: Tensor,
        batch_size: int,
        subproblem_state: Optional[SubproblemState],
        device: torch.device,
        propagate_preceeding_callback: Optional[
            Callable[
                [BacksubstitutionConfig, MN_BaB_Shape, bool],
                Tuple[MN_BaB_Shape, Tuple[Tensor, Tensor]],
            ]
        ],
        preceeding_layers: Optional[List[Any]],
        optimize_intermediate_bounds: bool,  # = False
        only_recompute_unstable: bool = False,  # = True for forward pass
    ) -> None:
        current_layer = self.layers[current_layer_index]
        prev_layer = self.get_prev_layer(current_layer_index, preceeding_layers)

        if prev_layer is None:
            # TODO: check validity before removing assertion
            assert current_layer.input_dim == input_lb.shape[1:]
            current_layer.update_input_bounds((input_lb, input_ub))
            return

        if (  # TODO: move this entire logic into Constraints?
            not optimize_intermediate_bounds
            and subproblem_state is not None
            and subproblem_state.constraints.split_state is not None
            and subproblem_state.constraints.layer_bounds is not None
            and layer_tag(current_layer)
            in subproblem_state.constraints.layer_bounds.intermediate_bounds
            and isinstance(current_layer, ReLU)
        ):
            unstable_nodes_in_current_layer = (
                subproblem_state.constraints.split_state.unstable_node_mask_in_layer(
                    layer_tag(current_layer)
                ).any(dim=0)
            )
            intermediate_bounds_to_recompute = unstable_nodes_in_current_layer.flatten()
        else:
            if only_recompute_unstable and current_layer.input_bounds is not None:
                intermediate_bounds_to_recompute = (
                    current_layer.input_bounds[0] * current_layer.input_bounds[1] < 0
                ).flatten()
            else:
                intermediate_bounds_to_recompute = torch.ones(
                    prev_layer.output_dim,
                    device=device,
                    dtype=torch.bool,
                ).flatten()

        if not intermediate_bounds_to_recompute.any():
            assert (
                subproblem_state is not None
                and layer_tag(current_layer)
                in subproblem_state.constraints.layer_bounds.intermediate_bounds
            )

            current_layer.update_input_bounds(
                subproblem_state.constraints.layer_bounds.intermediate_bounds[
                    layer_tag(current_layer)
                ]
            )
            return

        use_dependence_sets_for_current_bounds = (
            config.use_dependence_sets
            and self.layers[current_layer_index].dependence_set_applicable
        )
        if use_dependence_sets_for_current_bounds:
            config = config.without_prima()

        initial_intermediate_bound_coef_iter = (
            get_output_bound_initial_query_coef_iterator(
                dim=prev_layer.output_dim,
                intermediate_bounds_to_recompute=intermediate_bounds_to_recompute,
                use_dependence_sets=use_dependence_sets_for_current_bounds,
                batch_size=batch_size,
                slice_size=config.max_num_queries,
                device=device,
                dtype=None,  # TODO: should this be something else?
            )
        )

        subproblem_state_for_bounds = subproblem_state
        if subproblem_state_for_bounds is not None:
            if optimize_intermediate_bounds:
                subproblem_state_for_bounds = (
                    subproblem_state_for_bounds.with_new_parameters()
                )
            if use_dependence_sets_for_current_bounds:
                subproblem_state_for_bounds = (
                    subproblem_state_for_bounds.without_prima()
                )  # TODO: get rid of this?

        def get_interm_bound_callback(
            max_num_queries: Optional[int] = None,
        ) -> Callable[
            [QueryCoef],
            Tuple[
                Optional[MN_BaB_Shape], Tuple[Tensor, Tensor]
            ],  # TODO: do we need a MN_BaB_Shape result?
        ]:
            def interm_bound_callback(
                query_coef: QueryCoef,
            ) -> Tuple[
                Optional[MN_BaB_Shape], Tuple[Tensor, Tensor]
            ]:  # TODO: do we need a MN_BaB_Shape result?
                """
                Returns an abstract shape together with intermediate bounds after backpropagation.
                Abstract shape is None

                Args:
                    query_coef: all query coefficients
                """

                def get_interm_bound_restricted_queries(
                    query_coef: QueryCoef,
                    override_query_id: Optional[QueryTag] = None,
                ) -> Tuple[
                    Optional[MN_BaB_Shape], Tuple[Tensor, Tensor]
                ]:  # TODO: do we need a MN_BaB_Shape result?
                    if override_query_id is None:
                        query_id = query_tag(current_layer)
                    else:
                        query_id = override_query_id
                    abstract_shape = MN_BaB_Shape(
                        query_id=query_id,
                        query_prev_layer=prev_layer,
                        queries_to_compute=intermediate_bounds_to_recompute,  # TODO: only pass this if we will need it?
                        lb=AffineForm(query_coef),
                        ub=AffineForm(query_coef),
                        unstable_queries=None,  # initialized lazily if we will need it
                        subproblem_state=subproblem_state_for_bounds,
                    )
                    propagated_shape, layer_bounds = self._get_mn_bab_shape_after_layer(
                        from_layer_index=current_layer_index - 1,
                        config=config,
                        input_lb=input_lb,
                        input_ub=input_ub,
                        abstract_shape=abstract_shape,
                        propagate_preceeding_callback=propagate_preceeding_callback,
                        preceeding_layers=preceeding_layers,
                        use_early_termination_for_current_query=config.use_early_termination
                        if isinstance(current_layer, ReLU)
                        else False,
                        optimize_intermediate_bounds=optimize_intermediate_bounds,
                    )
                    if propagated_shape is not None:
                        assert layer_bounds is None
                        (
                            recomputed_layer_lb,
                            recomputed_layer_ub,
                        ) = propagated_shape.concretize(input_lb, input_ub)
                        assert recomputed_layer_ub is not None
                        if isinstance(current_layer, MaxPool2d):
                            recomputed_layer_lb = recomputed_layer_lb.view(
                                -1, *current_layer.input_dim
                            )
                            recomputed_layer_ub = recomputed_layer_ub.view(
                                -1, *current_layer.input_dim
                            )
                    else:
                        assert layer_bounds is not None
                        recomputed_layer_lb, recomputed_layer_ub = layer_bounds
                    return (
                        propagated_shape,
                        (recomputed_layer_lb, recomputed_layer_ub),
                    )

                def get_interm_bounds_in_multiple_query_groups(
                    query_coef: QueryCoef,
                    queries_per_group: int,
                    get_query_id: Callable[[int], QueryTag],
                ) -> Tuple[
                    Optional[MN_BaB_Shape], Tuple[Tensor, Tensor]
                ]:  # TODO: do we need a MN_BaB_Shape result?
                    total_queries = num_queries(query_coef)
                    device = query_coef.device
                    final_lbs = torch.zeros((batch_size, total_queries), device=device)
                    final_ubs = torch.zeros((batch_size, total_queries), device=device)
                    offset = 0
                    while offset < total_queries:
                        curr_end = min(offset + queries_per_group, total_queries)
                        curr_query_coef = query_coef[:, offset:curr_end]
                        prop_shape, (
                            curr_lb,
                            curr_ub,
                        ) = get_interm_bound_restricted_queries(
                            curr_query_coef, override_query_id=get_query_id(offset)
                        )
                        final_lbs[:, offset:curr_end] = curr_lb
                        final_ubs[:, offset:curr_end] = curr_ub
                        offset = curr_end
                    return prop_shape, (
                        final_lbs,
                        final_ubs,
                    )  # TODO: returning the final prop_shape seems a bit weird, why return a shape at all?

                # if (
                #     config.parameter_sharing_config is not None
                #     and subproblem_state_for_bounds is not None
                # ):  # TODO: clean up
                #     for (
                #         layer_type,
                #         sharing_config,
                #     ) in config.parameter_sharing_config.entries:
                #         if is_layer_of_type(prev_layer, layer_type):
                #             if sharing_config == ParameterSharing.same_layer:
                #                 break  # default behavior
                #             if sharing_config == ParameterSharing.none:
                #                 assert isinstance(
                #                     initial_intermediate_bound_coef, Tensor
                #                 )
                #                 return get_interm_bounds_in_multiple_query_groups(
                #                     query_coef=initial_intermediate_bound_coef,
                #                     queries_per_group=1,
                #                     get_query_id=lambda offset: query_tag_for_neuron(
                #                         current_layer, (offset,)
                #                     ),  # TODO: pass an index of the right shape
                #                 )
                #             if sharing_config == ParameterSharing.in_channel:
                #                 raise Exception(
                #                     "sharing parameters inside each channel not supported yet"
                #                 )

                if max_num_queries is None:
                    return get_interm_bound_restricted_queries(query_coef)
                else:
                    # assert isinstance(query_coef, Tensor)
                    query_id = query_tag(current_layer)
                    return get_interm_bounds_in_multiple_query_groups(
                        query_coef=query_coef,
                        queries_per_group=max_num_queries,
                        get_query_id=lambda offset: query_id,
                    )

            return interm_bound_callback

        if optimize_intermediate_bounds:
            pass
            # assert isinstance(initial_intermediate_bound_coef, Tensor)
            # assert subproblem_state_for_bounds is not None
            # (
            #     recomputed_layer_lb,
            #     recomputed_layer_ub,
            # ) = optimize_params_for_interm_bounds(
            #     query_id=query_tag(current_layer),
            #     query_coef=initial_intermediate_bound_coef,
            #     subproblem_state=subproblem_state_for_bounds,
            #     opt_callback=get_interm_bound_callback(config.max_num_queries),
            #     config=config.intermediate_bound_optimization_config,
            #     timeout=torch.inf,  # TODO: this is probably not right
            # )
            # recomputed_layer_lb, recomputed_layer_ub = (
            #     recomputed_layer_lb.detach(),
            #     recomputed_layer_ub.detach(),
            # )
            # if current_layer.optim_input_bounds is None:
            #     current_layer.optim_input_bounds = (
            #         recomputed_layer_lb,
            #         recomputed_layer_ub,
            #     )
            # else:
            #     current_layer.optim_input_bounds = (
            #         torch.max(recomputed_layer_lb, current_layer.optim_input_bounds[0]),
            #         torch.min(recomputed_layer_ub, current_layer.optim_input_bounds[1]),
            #     )
        else:

            interm_bound_callback = get_interm_bound_callback(config.max_num_queries)

            num_recompute = int(intermediate_bounds_to_recompute.float().sum().item())
            recomputed_layer_lb = torch.zeros(
                (batch_size, num_recompute), device=device
            )
            recomputed_layer_ub = torch.zeros(
                (batch_size, num_recompute), device=device
            )

            for slice_start, slice_end, coef in initial_intermediate_bound_coef_iter:
                debugging_shape, (
                    recomputed_layer_lb_slice,
                    recomputed_layer_ub_slice,
                ) = interm_bound_callback(coef)

                recomputed_layer_lb[
                    :, slice_start:slice_end
                ] = recomputed_layer_lb_slice.view((batch_size, -1))
                recomputed_layer_ub[
                    :, slice_start:slice_end
                ] = recomputed_layer_ub_slice.view((batch_size, -1))

        if not intermediate_bounds_to_recompute.all():
            assert (
                subproblem_state is not None
                and layer_tag(current_layer)
                in subproblem_state.constraints.layer_bounds.intermediate_bounds
            )

            (
                layer_lb,
                layer_ub,
            ) = subproblem_state.constraints.layer_bounds.intermediate_bounds[
                layer_tag(current_layer)
            ]
            layer_lb = layer_lb.flatten(start_dim=1).clone()
            layer_ub = layer_ub.flatten(start_dim=1).clone()

            layer_lb[:, intermediate_bounds_to_recompute] = recomputed_layer_lb
            layer_ub[:, intermediate_bounds_to_recompute] = recomputed_layer_ub
        else:
            layer_lb, layer_ub = recomputed_layer_lb, recomputed_layer_ub

        if (
            subproblem_state is not None
            and layer_tag(current_layer)
            in subproblem_state.constraints.layer_bounds.intermediate_bounds
        ):
            (
                best_lb,
                best_ub,
            ) = subproblem_state.constraints.layer_bounds.intermediate_bounds[
                layer_tag(current_layer)
            ]
            layer_lb = LeakyGradientMaximumFunction.apply(
                layer_lb, best_lb.view_as(layer_lb)
            )
            layer_ub = LeakyGradientMinimumFunction.apply(
                layer_ub, best_ub.view_as(layer_ub)
            )

        # print(f"lb_sum: {layer_lb.sum()}, ub_sum: {layer_ub.sum()}")
        current_layer.update_input_bounds((layer_lb, layer_ub), check_feasibility=True)
        if (
            isinstance(current_layer, ReLU)
            and subproblem_state is not None
            and subproblem_state.constraints.split_state is not None
        ):
            assert current_layer.input_bounds is not None
            subproblem_state.constraints.split_state.refine_split_constraints_for_relu(
                layer_tag(current_layer), current_layer.input_bounds
            )

    def _get_intermediate_bounds_callback(
        self,
        layer_index: int,
        config: BacksubstitutionConfig,
        subproblem_state: Optional[SubproblemState],
        device: torch.device,
        input_lb: Tensor,
        input_ub: Tensor,
        propagate_preceeding_callback: Optional[
            Callable[
                [BacksubstitutionConfig, MN_BaB_Shape, bool],
                Tuple[MN_BaB_Shape, Tuple[Tensor, Tensor]],
            ]
        ],
        preceeding_layers: Optional[List[Any]],
    ) -> Callable[[Tensor], Tuple[Tensor, Tensor]]:
        assert layer_index >= 1
        layer_input_shape = self.layers[layer_index - 1].output_dim
        current_layer = self.layers[layer_index]

        subproblem_state_for_queries = subproblem_state

        use_dependent_sets = config.use_dependence_sets

        if (
            subproblem_state_for_queries is not None and config.reduce_parameter_sharing
        ):  # use plain deep poly pass if reduced parameter sharing is active (there aren't canonical parameters to use and there is no optimization)
            subproblem_state_for_queries = (
                subproblem_state_for_queries.without_parameters()
            )

        @torch.no_grad()
        def compute_intermediate_bounds(
            query_coef: Tensor,
        ) -> Tuple[Tensor, Tensor]:
            query_coef = query_coef.view(
                *(query_coef.shape[:2] + layer_input_shape)
            ).to(device)

            query_shape = MN_BaB_Shape(  # Here AffineForm will be cloned later
                query_id=query_tag(current_layer),
                query_prev_layer=None,  # (not using reduced parameter sharing)
                queries_to_compute=None,  # we are given a complete set of queries
                lb=AffineForm(query_coef),
                ub=AffineForm(query_coef),
                unstable_queries=None,  # (see use_early_termination_for_current_query=False below)
                subproblem_state=subproblem_state_for_queries,
            )

            use_dependence_sets_for_current_bounds = (
                use_dependent_sets and current_layer.dependence_set_applicable
            )

            propagated_shape, __ = self._get_mn_bab_shape_after_layer(
                layer_index - 1,
                (
                    config.without_prima()
                    if use_dependence_sets_for_current_bounds
                    else config
                ).where(
                    use_early_termination=False,
                    layer_ids_for_which_to_compute_prima_constraints=[],
                ),
                input_lb,
                input_ub,
                query_shape,
                propagate_preceeding_callback=propagate_preceeding_callback,
                preceeding_layers=preceeding_layers,
                use_early_termination_for_current_query=False,  # This is only used for PRIMA constraints, there we almost only call this on queries that should be unstable
                optimize_intermediate_bounds=False,  # (intermediate bounds are optimized during the top level pass)
            )
            # assert query_shape.carried_over_optimizable_parameters == abstract_shape.carried_over_optimizable_parameters
            assert propagated_shape is not None
            ret_lbs, ret_ubs = propagated_shape.concretize(input_lb, input_ub)
            assert ret_ubs is not None
            return (ret_lbs, ret_ubs)

        return compute_intermediate_bounds

    def get_default_split_constraints(
        self, batch_size: int, device: torch.device
    ) -> Dict[LayerTag, Tensor]:
        split_constraints: Dict[LayerTag, Tensor] = {}
        for layer in self.layers:
            if (
                isinstance(layer, ReLU)
                or isinstance(layer, Sigmoid)
                or isinstance(layer, Tanh)
            ):
                split_constraints[layer_tag(layer)] = torch.zeros(
                    batch_size, *layer.output_dim, dtype=torch.int8, device=device
                )
            elif isinstance(layer, AbstractContainerModule):
                split_constraints.update(
                    layer.get_default_split_constraints(batch_size, device)
                )

        return split_constraints

    def get_default_split_points(
        self, batch_size: int, device: torch.device
    ) -> Dict[LayerTag, Tensor]:
        split_points: Dict[LayerTag, Tensor] = {}
        for layer in self.layers:
            if isinstance(layer, Sigmoid) or isinstance(layer, Tanh):
                split_points[layer_tag(layer)] = torch.zeros(
                    batch_size, *layer.output_dim, dtype=torch.float32, device=device
                )
            elif isinstance(layer, AbstractContainerModule):
                split_points.update(layer.get_default_split_points(batch_size, device))

        return split_points

    def get_activation_layers(self) -> Dict[LayerTag, ActivationLayer]:
        act_layers: Dict[LayerTag, ActivationLayer] = {}
        for layer in self.layers:
            if (
                isinstance(layer, ReLU)
                or isinstance(layer, Sigmoid)
                or isinstance(layer, Tanh)
            ):
                act_layers[layer_tag(layer)] = layer
            elif isinstance(layer, AbstractContainerModule):
                act_layers.update(layer.get_activation_layers())

        return act_layers

    def get_current_intermediate_bounds(
        self,
    ) -> OrderedDict[LayerTag, Tuple[Tensor, Tensor]]:
        intermediate_bounds: OrderedDict[
            LayerTag, Tuple[Tensor, Tensor]
        ] = OrderedDict()
        for layer in self.layers:
            if layer.input_bounds is not None:
                intermediate_bounds[layer_tag(layer)] = (
                    layer.input_bounds[0].detach(),
                    layer.input_bounds[1].detach(),
                )
            if isinstance(layer, AbstractContainerModule):
                intermediate_bounds.update(layer.get_current_intermediate_bounds())
        return intermediate_bounds

    def get_current_optimized_intermediate_bounds(
        self,
    ) -> OrderedDict[LayerTag, Tuple[Tensor, Tensor]]:
        intermediate_bounds: OrderedDict[
            LayerTag, Tuple[Tensor, Tensor]
        ] = OrderedDict()
        for layer in self.layers:
            if layer.optim_input_bounds is not None:
                intermediate_bounds[layer_tag(layer)] = (
                    layer.optim_input_bounds[0].detach(),
                    layer.optim_input_bounds[1].detach(),
                )
            if isinstance(layer, AbstractContainerModule):
                intermediate_bounds.update(
                    layer.get_current_optimized_intermediate_bounds()
                )
        return intermediate_bounds

    def get_babsr_bias(self, from_layer_index: Optional[int] = None) -> Tensor:
        if from_layer_index is None:
            from_layer_index = len(self.layers) - 1
        for i, layer in reversed(list(enumerate(self.layers[: from_layer_index + 1]))):
            if isinstance(layer, Sequential):
                return layer.get_babsr_bias()
            elif hasattr(layer, "bias"):
                return layer.bias
        # nn.Parameter so it is automatically moved to correct device
        # and converted to correct dtype in nn.Module constructor
        return nn.Parameter(torch.zeros((1,)))

    def get_activation_layer_ids(
        self, act_layer_ids: Optional[List[LayerTag]] = None
    ) -> List[LayerTag]:
        if act_layer_ids is None:
            act_layer_ids = []
        for layer in self.layers:
            act_layer_ids += layer.get_activation_layer_ids()
        return act_layer_ids

    def get_relu_layer_ids(
        self, act_layer_ids: Optional[List[LayerTag]] = None
    ) -> List[LayerTag]:
        if act_layer_ids is None:
            act_layer_ids = []
        for layer in self.layers:
            act_layer_ids += layer.get_relu_layer_ids()
        return act_layer_ids

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

        device = input_lb.device
        inner_preceeding_layers: List[AbstractModule] = []
        tracked_preceeding_layers: List[AbstractModule] = []
        inner_propagate_preceeding_callback = propagate_preceeding_callback
        ibp_call()

        for i, layer in list(enumerate(self.layers)):
            if time.time() > timeout:
                return
            if type(layer) in ActivationLayers:

                subproblem_state = SubproblemState.create_default(
                    split_state=None,
                    optimize_prima=False,
                    batch_size=1,
                    device=input_lb.device,
                    use_params=False,
                )

                subproblem_state.constraints.layer_bounds.intermediate_bounds = (
                    self.get_current_intermediate_bounds()
                )
                pre_unstable = (
                    (layer.input_bounds[0] * layer.input_bounds[1] < 0).float().sum()
                )
                pre_width = (layer.input_bounds[1] - layer.input_bounds[0]).mean()

                if pre_unstable > 0:
                    # Here the esequential takes care of preceeding layers
                    self._set_intermediate_bounds(
                        current_layer_index=i,
                        config=config,
                        input_lb=input_lb,
                        input_ub=input_ub,
                        batch_size=subproblem_state.batch_size,
                        subproblem_state=subproblem_state,
                        device=device,
                        propagate_preceeding_callback=propagate_preceeding_callback,
                        preceeding_layers=preceeding_layers,
                        optimize_intermediate_bounds=False,
                        only_recompute_unstable=True,
                    )
                    ibp_call()

                post_unstable = (
                    (layer.input_bounds[0] * layer.input_bounds[1] < 0).float().sum()
                )
                post_width = (layer.input_bounds[1] - layer.input_bounds[0]).mean()
                print(
                    f"ID: {id(layer)} | Pre: {pre_unstable:.0f} | Post: {post_unstable:.0f} | Pre-Width: {pre_width} | Post-Width: {post_width} | TR {timeout - time.time():.3f}"
                )

            if isinstance(layer, AbstractContainerModule):

                # Updated preceeding callback
                # Callback for all layers before the current AbstractContainerModule
                if len(inner_preceeding_layers) > 0:
                    inner_propagate_preceeding_callback = (
                        self._get_preceeding_callback_wrapper(
                            inner_propagate_preceeding_callback, inner_preceeding_layers
                        )
                    )

                if preceeding_layers is None:
                    temp_preceeding_layers = [tracked_preceeding_layers]
                else:
                    temp_preceeding_layers = [
                        *preceeding_layers,
                        tracked_preceeding_layers,
                    ]

                layer.forward_pass(
                    config=config,
                    input_lb=input_lb,
                    input_ub=input_ub,
                    propagate_preceeding_callback=inner_propagate_preceeding_callback,
                    preceeding_layers=temp_preceeding_layers,
                    ibp_call=ibp_call,
                    timeout=timeout,
                )

                # Callback including the current AbstractContainerModule
                def get_preceeding_callback(
                    layer: AbstractContainerModule,
                    existing_preceeding_callback: Optional[
                        Callable[
                            [
                                BacksubstitutionConfig,
                                MN_BaB_Shape,
                                bool,
                            ],
                            Tuple[MN_BaB_Shape, Tuple[Tensor, Tensor]],
                        ]
                    ],
                ) -> Callable[
                    [BacksubstitutionConfig, MN_BaB_Shape, bool],
                    Tuple[MN_BaB_Shape, Tuple[Tensor, Tensor]],
                ]:
                    def _propagate_preceeding_callback(
                        config: BacksubstitutionConfig,
                        abstract_shape_int: MN_BaB_Shape,
                        use_early_termination_for_current_query: bool,
                    ) -> Tuple[MN_BaB_Shape, Tuple[Tensor, Tensor]]:
                        (abstract_shape_int, (lbs, ubs),) = layer.backsubstitute_shape(
                            config=config,
                            input_lb=input_lb,
                            input_ub=input_ub,
                            abstract_shape=abstract_shape_int,
                            from_layer_index=i,
                            propagate_preceeding_callback=None,
                            preceeding_layers=temp_preceeding_layers,  # Which layers are required here?
                            use_early_termination_for_current_query=use_early_termination_for_current_query,
                            full_back_prop=False,
                            optimize_intermediate_bounds=False,
                        )

                        if existing_preceeding_callback is not None:
                            return existing_preceeding_callback(
                                config,
                                abstract_shape_int,
                                use_early_termination_for_current_query,
                            )
                        else:
                            return (
                                abstract_shape_int,
                                (lbs, ubs),
                            )

                    return _propagate_preceeding_callback

                inner_propagate_preceeding_callback = get_preceeding_callback(
                    layer, inner_propagate_preceeding_callback
                )
                inner_preceeding_layers = []
                ibp_call()

            else:
                inner_preceeding_layers.append(layer)
            tracked_preceeding_layers.append(layer)

    def _get_preceeding_callback_wrapper(
        self,
        propagate_preceeding_callback: Optional[
            Callable[
                [BacksubstitutionConfig, MN_BaB_Shape, bool],
                Tuple[MN_BaB_Shape, Tuple[Tensor, Tensor]],
            ]
        ],
        layers: List[AbstractModule],
    ) -> Callable[
        [BacksubstitutionConfig, MN_BaB_Shape, bool],
        Tuple[MN_BaB_Shape, Tuple[Tensor, Tensor]],
    ]:
        def wrapped_call(
            config: BacksubstitutionConfig,
            abstract_shape: MN_BaB_Shape,
            use_early_termination_for_current_query: bool,
        ) -> Tuple[MN_BaB_Shape, Tuple[Tensor, Tensor]]:

            for layer in reversed(layers):
                abstract_shape = layer.backsubstitute(config, abstract_shape)

            if propagate_preceeding_callback is None:
                # assert isinstance(abstract_shape.lb.coef, Tensor)
                # bound_shape = abstract_shape.lb.coef.shape[:2]
                # if isinstance(abstract_shape.lb.coef, Tensor):
                #     bound_shape = abstract_shape.lb.coef.shape[:2]
                # elif isinstance(abstract_shape.lb.coef, DependenceSets):
                #     bound_shape = abstract_shape.lb.coef.sets.shape[:2]
                # else:
                #     assert False
                return (
                    abstract_shape,
                    (
                        -np.inf * torch.ones_like(abstract_shape.lb.bias),
                        np.inf * torch.ones_like(abstract_shape.lb.bias),
                    ),
                )
            else:
                return propagate_preceeding_callback(
                    config,
                    abstract_shape,
                    use_early_termination_for_current_query,
                )

        return wrapped_call
