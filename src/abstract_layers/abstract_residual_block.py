from __future__ import annotations

from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from src.abstract_domains.ai_util import AbstractElement
from src.abstract_layers.abstract_container_module import (
    AbstractContainerModule,
    ActivationLayer,
)
from src.abstract_layers.abstract_sequential import Sequential
from src.concrete_layers import basic_block as concrete_basic_block
from src.concrete_layers.residual_block import ResidualBlock as concreteResidualBlock
from src.mn_bab_shape import AffineForm, MN_BaB_Shape
from src.state.tags import LayerTag
from src.utilities.config import BacksubstitutionConfig
from src.utilities.dependence_sets import DependenceSets
from src.utilities.queries import QueryCoef
from src.verification_subproblem import SubproblemState


class ResidualBlock(concreteResidualBlock, AbstractContainerModule):
    path_a: Sequential  # type: ignore[assignment] # hack
    path_b: Sequential  # type: ignore[assignment]

    def __init__(
        self,
        path_a: nn.Sequential,
        path_b: nn.Sequential,
        input_dim: Tuple[int, ...],
        **kwargs: Any,
    ) -> None:
        super(ResidualBlock, self).__init__(path_a=path_a, path_b=path_b)
        self.path_a = Sequential.from_concrete_module(path_a, input_dim, **kwargs)
        self.path_b = Sequential.from_concrete_module(path_b, input_dim, **kwargs)
        self.output_dim = self.path_b.layers[-1].output_dim
        self.input_dim = input_dim
        self.bias = self.get_babsr_bias()

    @classmethod
    def from_concrete_module(  # type: ignore[override] # (checked at runtime)
        cls,
        module: concrete_basic_block.ResidualBlock,
        input_dim: Tuple[int, ...],
        **kwargs: Any,
    ) -> ResidualBlock:
        assert isinstance(module, concrete_basic_block.ResidualBlock)
        abstract_layer = cls(
            module.path_a,
            module.path_b,
            input_dim,
            **kwargs,
        )
        return abstract_layer

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
        in_lb = abstract_shape.lb.clone()  # Also works properly for dependence set
        in_ub: Optional[AffineForm] = None
        if abstract_shape.ub is not None:
            in_ub = abstract_shape.ub.clone()

        unstable_queries_old_for_assert = abstract_shape.unstable_queries

        (a_shape_a, (lbs_a, ubs_a),) = self.path_a.backsubstitute_shape(
            config=config,
            input_lb=input_lb,
            input_ub=input_ub,
            abstract_shape=abstract_shape,
            from_layer_index=None,
            propagate_preceeding_callback=propagate_preceeding_callback,
            preceeding_layers=preceeding_layers,
            use_early_termination_for_current_query=False,
            full_back_prop=False,
            optimize_intermediate_bounds=optimize_intermediate_bounds,
        )
        assert (
            abstract_shape.unstable_queries is None
            or (
                abstract_shape.unstable_queries == unstable_queries_old_for_assert
            ).all()
        )

        a_lb = a_shape_a.lb.clone()
        if a_shape_a.ub is not None:
            a_ub = a_shape_a.ub.clone()

        abstract_shape.update_bounds(in_lb, in_ub)
        a_shape_b, __ = self.path_b.backsubstitute_shape(
            config=config,
            input_lb=input_lb,
            input_ub=input_ub,
            abstract_shape=abstract_shape,
            from_layer_index=None,
            propagate_preceeding_callback=propagate_preceeding_callback,
            preceeding_layers=preceeding_layers,
            use_early_termination_for_current_query=False,
            full_back_prop=False,
            optimize_intermediate_bounds=optimize_intermediate_bounds,
        )
        assert (
            abstract_shape.unstable_queries is None
            or (
                abstract_shape.unstable_queries == unstable_queries_old_for_assert
            ).all()
        )

        new_lower: AffineForm
        new_upper: Optional[AffineForm] = None

        new_lb_coef: QueryCoef
        new_lb_bias = (
            a_lb.bias + a_shape_b.lb.bias - in_lb.bias
        )  # Both the shape in a and in b  contain the initial bias terms, so one has to be subtracted
        if isinstance(a_lb.coef, DependenceSets) and not isinstance(
            a_shape_b.lb.coef, DependenceSets
        ):
            new_lb_coef = (
                a_lb.coef.to_tensor(a_shape_b.lb.coef.shape[-3:]) + a_shape_b.lb.coef
            )
        elif not isinstance(a_lb.coef, DependenceSets) and isinstance(
            a_shape_b.lb.coef, DependenceSets
        ):
            new_lb_coef = a_lb.coef + a_shape_b.lb.coef.to_tensor(a_lb.coef.shape[-3:])
        else:
            new_lb_coef = a_lb.coef + a_shape_b.lb.coef

        new_lower = AffineForm(new_lb_coef, new_lb_bias)

        if a_shape_b.ub is not None and a_ub is not None and in_ub is not None:
            new_ub_coef: QueryCoef
            new_ub_bias = a_ub.bias + a_shape_b.ub.bias - in_ub.bias
            if isinstance(a_ub.coef, DependenceSets) and not isinstance(
                a_shape_b.ub.coef, DependenceSets
            ):
                new_ub_coef = (
                    a_ub.coef.to_tensor(a_shape_b.ub.coef.shape[-3:])
                    + a_shape_b.ub.coef
                )
            elif not isinstance(a_ub.coef, DependenceSets) and isinstance(
                a_shape_b.ub.coef, DependenceSets
            ):
                new_ub_coef = a_ub.coef + a_shape_b.ub.coef.to_tensor(
                    a_ub.coef.shape[-3:]
                )
            else:
                new_ub_coef = a_ub.coef + a_shape_b.ub.coef
            new_upper = AffineForm(new_ub_coef, new_ub_bias)

        abstract_shape.update_bounds(new_lower, new_upper)
        return (
            abstract_shape,
            (
                -np.inf * torch.ones_like(lbs_a),
                np.inf * torch.ones_like(lbs_a),
            ),  # TODO: this seems unnecessary, move bounds into abstract_shape and just update them when it makes sense
        )

    def get_babsr_bias(self) -> Tensor:
        bias_a = self.path_a.get_babsr_bias()
        bias_b = self.path_b.get_babsr_bias()
        # In case one of them is one the gpu we will move both to gpu
        # Have to do this here as the paths (sequentials) are unaware of the device
        if bias_a.is_cuda != bias_b.is_cuda:
            bias_a = bias_a.cuda()
            bias_b = bias_b.cuda()
        return nn.Parameter(bias_a + bias_b)

    def reset_input_bounds(self) -> None:
        super(ResidualBlock, self).reset_input_bounds()
        self.path_a.reset_input_bounds()
        self.path_b.reset_input_bounds()

    def reset_optim_input_bounds(self) -> None:
        super(ResidualBlock, self).reset_input_bounds()
        self.path_a.reset_optim_input_bounds()
        self.path_b.reset_optim_input_bounds()

    def reset_output_bounds(self) -> None:
        super(ResidualBlock, self).reset_output_bounds()
        self.path_a.reset_output_bounds()
        self.path_b.reset_output_bounds()

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
        self.path_a.forward_pass(
            config,
            input_lb,
            input_ub,
            propagate_preceeding_callback,
            preceeding_layers,
            ibp_call,
            timeout,
        )
        self.path_b.forward_pass(
            config,
            input_lb,
            input_ub,
            propagate_preceeding_callback,
            preceeding_layers,
            ibp_call,
            timeout,
        )

    def propagate_interval(
        self,
        interval: Tuple[Tensor, Tensor],
        use_existing_bounds: Optional[bool] = None,
        subproblem_state: Optional[SubproblemState] = None,
        activation_layer_only: bool = False,
        set_input: bool = True,
        set_output: bool = True,
    ) -> Tuple[Tensor, Tensor]:
        interval_a = self.path_a.propagate_interval(
            interval,
            use_existing_bounds,
            subproblem_state,
            activation_layer_only=activation_layer_only,
            set_input=set_input,
            set_output=set_output,
        )
        interval_b = self.path_b.propagate_interval(
            interval,
            use_existing_bounds,
            subproblem_state,
            activation_layer_only=activation_layer_only,
            set_input=set_input,
            set_output=set_output,
        )

        return interval_a[0] + interval_b[0], interval_a[1] + interval_b[1]

    def propagate_abstract_element(
        self,
        abs_input: AbstractElement,
        use_existing_bounds: Optional[bool] = None,
        activation_layer_only: bool = False,
        set_input: bool = True,
        set_output: bool = True,
    ) -> AbstractElement:
        abs_output_a = self.path_a.propagate_abstract_element(
            abs_input,
            use_existing_bounds,
            activation_layer_only,
            set_input=set_input,
            set_output=set_output,
        )
        abs_output_b = self.path_b.propagate_abstract_element(
            abs_input,
            use_existing_bounds,
            activation_layer_only,
            set_input=set_input,
            set_output=set_output,
        )
        return abs_output_a + abs_output_b

    def set_dependence_set_applicability(self, applicable: bool = True) -> None:
        self.path_a.set_dependence_set_applicability(applicable)
        self.path_b.set_dependence_set_applicability(applicable)
        self.dependence_set_applicable = (
            self.path_a.layers[-1].dependence_set_applicable
            and self.path_b.layers[-1].dependence_set_applicable
        )

    def get_default_split_constraints(
        self, batch_size: int, device: torch.device
    ) -> Dict[LayerTag, Tensor]:
        split_constraints: Dict[LayerTag, Tensor] = {}
        split_constraints.update(
            self.path_a.get_default_split_constraints(batch_size, device)
        )
        split_constraints.update(
            self.path_b.get_default_split_constraints(batch_size, device)
        )
        return split_constraints

    def get_default_split_points(
        self, batch_size: int, device: torch.device
    ) -> Dict[LayerTag, Tensor]:
        split_points: Dict[LayerTag, Tensor] = {}
        split_points.update(self.path_a.get_default_split_points(batch_size, device))
        split_points.update(self.path_b.get_default_split_points(batch_size, device))
        return split_points

    def get_activation_layers(self) -> Dict[LayerTag, ActivationLayer]:
        act_layers: Dict[LayerTag, ActivationLayer] = {}
        act_layers.update(self.path_a.get_activation_layers())
        act_layers.update(self.path_b.get_activation_layers())
        return act_layers

    def get_current_intermediate_bounds(
        self,
    ) -> OrderedDict[LayerTag, Tuple[Tensor, Tensor]]:
        intermediate_bounds: OrderedDict[
            LayerTag, Tuple[Tensor, Tensor]
        ] = OrderedDict()
        intermediate_bounds.update(self.path_a.get_current_intermediate_bounds())
        intermediate_bounds.update(self.path_b.get_current_intermediate_bounds())
        return intermediate_bounds

    def get_current_optimized_intermediate_bounds(
        self,
    ) -> OrderedDict[LayerTag, Tuple[Tensor, Tensor]]:
        intermediate_bounds: OrderedDict[
            LayerTag, Tuple[Tensor, Tensor]
        ] = OrderedDict()
        intermediate_bounds.update(
            self.path_a.get_current_optimized_intermediate_bounds()
        )
        intermediate_bounds.update(
            self.path_b.get_current_optimized_intermediate_bounds()
        )
        return intermediate_bounds

    def set_intermediate_input_bounds(
        self, intermediate_bounds: OrderedDict[LayerTag, Tuple[Tensor, Tensor]]
    ) -> None:
        self.path_a.set_intermediate_input_bounds(intermediate_bounds)
        self.path_b.set_intermediate_input_bounds(intermediate_bounds)

    def get_activation_layer_ids(
        self, act_layer_ids: Optional[List[LayerTag]] = None
    ) -> List[LayerTag]:
        if act_layer_ids is None:
            act_layer_ids = []
        act_layer_ids += self.path_a.get_activation_layer_ids()
        act_layer_ids += self.path_b.get_activation_layer_ids()
        return act_layer_ids

    def get_relu_layer_ids(
        self, act_layer_ids: Optional[List[LayerTag]] = None
    ) -> List[LayerTag]:
        if act_layer_ids is None:
            act_layer_ids = []
        act_layer_ids += self.path_a.get_relu_layer_ids()
        act_layer_ids += self.path_b.get_relu_layer_ids()
        return act_layer_ids
