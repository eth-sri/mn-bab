from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple

import torch
from torch import Size, Tensor, nn

from src.abstract_domains.ai_util import AbstractElement
from src.abstract_layers.abstract_container_module import (
    ActivationLayer,
    ActivationLayers,
)
from src.abstract_layers.abstract_module import AbstractModule
from src.abstract_layers.abstract_sequential import Sequential
from src.mn_bab_shape import AffineForm, MN_BaB_Shape
from src.state.tags import LayerTag, QueryTag
from src.utilities.abstract_module_mapper import AbstractModuleMapper
from src.utilities.config import BacksubstitutionConfig
from src.verification_subproblem import SubproblemState


class AbstractNetwork(Sequential):
    def __init__(
        self,
        layers: Iterable[AbstractModule],
    ) -> None:
        super(AbstractNetwork, self).__init__(layers)
        self.layer_id_to_layer: Dict[LayerTag, ActivationLayer] = {}
        self.has_output_adapter: bool = False  # Whether or not we appended an adapter to the network for computing a disjunctive clause

    @classmethod
    def from_concrete_module(  # type: ignore[override] # (checked at runtime)
        cls, module: nn.Sequential, input_dim: Tuple[int, ...], **kwargs: Any
    ) -> AbstractNetwork:
        assert isinstance(module, nn.Sequential)
        layers = Sequential.from_concrete_module(
            module,
            input_dim,
            concrete_to_abstract=AbstractModuleMapper.map_to_abstract_type,
        ).layers

        obj = cls(layers)
        obj.set_activation_layers()
        return obj

    def append_out_adapter(
        self, module: nn.Sequential, device: torch.device, dtype: torch.dtype
    ) -> None:
        """
        Appends an output adapter to the current network, i.e.,
        we append several layer to the network to encode specific output properties

        Args:
            module (nn.Sequential): The sequential layer to append (represents the concrete layer)

        Returns:
            _type_: Reference to the inplace-updated abstract-network.
        """
        assert not self.has_output_adapter
        new_layers = (
            Sequential.from_concrete_module(
                module,
                self.output_dim,
                concrete_to_abstract=AbstractModuleMapper.map_to_abstract_type,
            )
            .to(device)
            .to(dtype)
        )

        # Note that we add it as a single sequential block
        self.has_output_adapter = True
        self.layers.append(new_layers)
        self.set_activation_layers()
        self.output_dim = new_layers.output_dim

    def remove_out_adapter(self) -> None:
        assert self.has_output_adapter
        self.layers = self.layers[:-1]
        self.has_output_adapter = False
        self.set_activation_layers()
        self.output_dim = self.layers[-1].output_dim

    def get_mn_bab_shape(
        self,
        config: BacksubstitutionConfig,
        input_lb: Tensor,
        input_ub: Tensor,
        query_id: QueryTag,
        query_coef: Tensor,
        subproblem_state: Optional[SubproblemState],
        compute_upper_bound: bool,
        reset_input_bounds: bool,  # = True  # TODO: is this ever False?
        optimize_intermediate_bounds: bool,  # = False
        recompute_intermediate_bounds: bool,
    ) -> MN_BaB_Shape:
        assert query_coef.is_leaf
        abstract_shape = MN_BaB_Shape(
            query_id=query_id,
            query_prev_layer=None,  # TODO: reduced parameter sharing for the output layer?
            queries_to_compute=None,  # compute all queries
            lb=AffineForm(query_coef),
            ub=AffineForm(query_coef) if compute_upper_bound else None,
            unstable_queries=None,  # (not using early termination)
            subproblem_state=subproblem_state,
        )
        return self.backsubstitute_mn_bab_shape(
            config=config,
            input_lb=input_lb,
            input_ub=input_ub,
            query_coef=None,
            abstract_shape=abstract_shape,
            compute_upper_bound=compute_upper_bound,
            reset_input_bounds=reset_input_bounds,
            optimize_intermediate_bounds=optimize_intermediate_bounds,
            recompute_intermediate_bounds=recompute_intermediate_bounds,
        )

    def backsubstitute_mn_bab_shape(  # TODO: get rid of this?
        self,
        config: BacksubstitutionConfig,
        input_lb: Tensor,
        input_ub: Tensor,
        query_coef: Optional[Tensor],
        abstract_shape: MN_BaB_Shape,
        compute_upper_bound: bool,
        reset_input_bounds: bool,  # = True  # TODO: is this ever False?
        recompute_intermediate_bounds: bool,
        optimize_intermediate_bounds: bool,  # = False
    ) -> MN_BaB_Shape:
        if reset_input_bounds:
            self.reset_input_bounds()
        subproblem_state = abstract_shape.subproblem_state
        if subproblem_state is not None:
            if recompute_intermediate_bounds:  # only set the bounds to be kept fix
                self.set_intermediate_input_bounds(
                    subproblem_state.constraints.layer_bounds.fixed_intermediate_bounds
                )
            else:  # Set all bounds => recompute none
                self.set_intermediate_input_bounds(
                    subproblem_state.constraints.layer_bounds.intermediate_bounds
                )

        if query_coef is not None:
            assert query_coef.is_leaf
            abstract_shape.update_bounds(  # Cloning necessary to prevent aliasing
                AffineForm(query_coef.clone()),
                AffineForm(query_coef.clone()) if compute_upper_bound else None,
            )

        shape, __ = self._get_mn_bab_shape_after_layer(
            from_layer_index=len(self.layers) - 1,
            config=config,
            input_lb=input_lb,
            input_ub=input_ub,
            abstract_shape=abstract_shape,
            propagate_preceeding_callback=None,
            preceeding_layers=None,
            use_early_termination_for_current_query=False,
            optimize_intermediate_bounds=optimize_intermediate_bounds,
        )
        assert shape is not None
        return shape

    def set_activation_layers(self) -> None:
        self.layer_id_to_layer = self.get_activation_layers()

    def activation_layer_bounds_to_optim_layer_bounds(self) -> None:
        act_layers = self.get_activation_layers()
        for id, layer in act_layers.items():
            if layer.optim_input_bounds is None:
                if layer.input_bounds is not None:
                    layer.optim_input_bounds = (
                        layer.input_bounds[0].detach(),
                        layer.input_bounds[1].detach(),
                    )
            elif layer.input_bounds is not None:
                opt_lb = torch.max(
                    layer.input_bounds[0].detach(), layer.optim_input_bounds[0]
                )
                opt_ub = torch.min(
                    layer.input_bounds[1].detach(), layer.optim_input_bounds[1]
                )
                layer.optim_input_bounds = (opt_lb, opt_ub)

    def set_layer_bounds_via_forward_dp_pass(
        self,
        config: BacksubstitutionConfig,
        input_lb: Tensor,
        input_ub: Tensor,
        timeout: float,
    ) -> None:

        # wraps layers in a sequential
        layers_as_seq = Sequential(self.layers)

        # Create a reference to trigger a forward ibp call

        def ibp_call() -> None:
            self.set_layer_bounds_via_interval_propagation(
                input_lb,
                input_ub,
                use_existing_bounds=True,
                activation_layer_only=True,
                has_batch_dim=True,
                set_input=True,
                set_output=False,
            )

        layers_as_seq.forward_pass(
            config=config,
            input_lb=input_lb,
            input_ub=input_ub,
            propagate_preceeding_callback=None,
            preceeding_layers=None,
            ibp_call=ibp_call,
            timeout=timeout,
        )
        # final_layer = layers_as_seq.layers[-1]
        # assert isinstance(final_layer, AbstractModule)
        # assert final_layer.output_bounds is not None
        # return final_layer.output_bounds

    def set_layer_bounds_via_interval_propagation(
        self,
        input_lb: Tensor,
        input_ub: Tensor,
        use_existing_bounds: bool = False,
        subproblem_state: Optional[SubproblemState] = None,
        activation_layer_only: bool = False,
        set_input: bool = True,
        set_output: bool = True,
        has_batch_dim: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        if not has_batch_dim:
            if len(input_lb.shape) in [2, 4]:
                shape_with_batch_dimension = input_lb.shape
            elif len(input_lb.shape) in [1, 3]:
                shape_with_batch_dimension = Size((1, *(input_lb.shape)))
            else:
                raise RuntimeError(
                    "Unexpected number of dimensions for interval propagation."
                )
            interval = (
                input_lb.expand(shape_with_batch_dimension),
                input_ub.expand(shape_with_batch_dimension),
            )
        else:
            interval = (input_lb, input_ub)
        for layer in self.layers:
            if use_existing_bounds and layer.input_bounds is not None:
                lb = torch.max(interval[0], layer.input_bounds[0].view_as(interval[0]))
                ub = torch.min(interval[1], layer.input_bounds[1].view_as(interval[1]))
                interval = (lb, ub)

            if use_existing_bounds and layer.optim_input_bounds is not None:
                lb = torch.max(
                    interval[0],
                    layer.optim_input_bounds[0].view(-1, *interval[0].shape[1:]),
                )
                ub = torch.min(
                    interval[1],
                    layer.optim_input_bounds[1].view(-1, *interval[1].shape[1:]),
                )
                interval = (lb, ub)

            if set_input and (
                type(layer) in ActivationLayers or not activation_layer_only
            ):
                layer.update_input_bounds(interval, check_feasibility=False)
            interval = layer.propagate_interval(
                interval,
                use_existing_bounds,
                subproblem_state,
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

    def set_layer_bounds_via_abstract_element_propagation(
        self,
        abs_input: AbstractElement,
        use_existing_bounds: bool = False,
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
