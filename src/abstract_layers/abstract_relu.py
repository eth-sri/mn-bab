from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from src.abstract_domains.ai_util import AbstractElement
from src.abstract_layers.abstract_conv2d import Conv2d
from src.abstract_layers.abstract_module import AbstractModule
from src.mn_bab_shape import AffineForm, MN_BaB_Shape
from src.state.tags import (
    LayerTag,
    ParameterTag,
    key_alpha_relu,
    key_beta,
    key_prima,
    key_prima_lb,
    key_prima_ub,
    layer_tag,
)
from src.utilities.config import (
    BacksubstitutionConfig,
    ParameterSharing,
    PrimaHyperparameters,
    ReLUAlphaInitMethod,
)
from src.utilities.dependence_sets import DependenceSets
from src.utilities.layer_types import is_layer_of_type
from src.utilities.leaky_gradient_maximum_function import LeakyGradientMaximumFunction
from src.utilities.leaky_gradient_minimum_function import LeakyGradientMinimumFunction
from src.utilities.prima_interface import ActivationType, get_prima_constraints
from src.verification_subproblem import SubproblemState

EPS = 1e-15


class ReLU(nn.ReLU, AbstractModule):
    def __init__(self, dim: Tuple[int, ...]) -> None:
        super(ReLU, self).__init__()
        self.output_dim = dim
        self.dependence_set_block = False

    @classmethod
    def from_concrete_module(  # type: ignore[override] # (checked at runtime)
        cls, module: nn.ReLU, input_dim: Tuple[int, ...], **kwargs: Any
    ) -> ReLU:
        assert isinstance(module, nn.ReLU)
        return cls(input_dim)

    def update_input_bounds(
        self, input_bounds: Tuple[Tensor, Tensor], check_feasibility: bool = True
    ) -> None:
        input_bounds_shape_adjusted = (
            input_bounds[0].view(-1, *self.output_dim),
            input_bounds[1].view(-1, *self.output_dim),
        )
        super(ReLU, self).update_input_bounds(
            input_bounds_shape_adjusted, check_feasibility=check_feasibility
        )

    def backsubstitute(
        self,
        config: BacksubstitutionConfig,
        abstract_shape: MN_BaB_Shape,
        intermediate_bounds_callback: Optional[
            Callable[[Tensor], Tuple[Tensor, Tensor]]
        ] = None,
        prev_layer: Optional[AbstractModule] = None,
    ) -> MN_BaB_Shape:
        if self.input_bounds is None:
            raise RuntimeError("Cannot backsubstitute if bounds have not been set.")

        # Shared computation of input bounds and prima coefficients
        if self.optim_input_bounds is not None:
            optim_lb = self.optim_input_bounds[0].view(1, *self.output_dim)
            optim_ub = self.optim_input_bounds[1].view(1, *self.output_dim)
            input_lb = LeakyGradientMaximumFunction.apply(
                self.input_bounds[0], optim_lb.broadcast_to(self.input_bounds[0].shape)
            )
            input_ub = LeakyGradientMinimumFunction.apply(
                self.input_bounds[1], optim_ub.broadcast_to(self.input_bounds[1].shape)
            )
            input_bounds = (input_lb, input_ub)
        else:
            input_bounds = self.input_bounds

        prima_coefs = self._get_prima_coefficients(
            config, abstract_shape, intermediate_bounds_callback
        )

        prima_constraints_available = (
            prima_coefs is not None and prima_coefs[0].shape[2] > 0
        )

        # NOTE: This changes the abstract shape in-place leading to consequences for all
        # further .matmaul or .elemwise calls on it Thus it needs to be called before
        # the ._backsub_affine_form calls()
        if prima_constraints_available:
            assert prima_coefs is not None
            assert not abstract_shape.uses_dependence_sets()
            (
                prima_output_coefs,
                prima_input_coefs,
                prima_const_coefs,
            ) = prima_coefs

            prima_parameter_shape = (
                abstract_shape.batch_size,
                prima_output_coefs.shape[2],
                1,
            )
            prima_lb_parameters = abstract_shape.get_parameters(
                key_prima_lb, layer_tag(self), prima_parameter_shape
            )

            abstract_shape.lb.coef += self._multiply_prima_coefs_and_parameters(
                prima_output_coefs, prima_lb_parameters
            )
            if abstract_shape.ub is not None:
                assert isinstance(abstract_shape.ub.coef, Tensor)
                prima_ub_parameters = abstract_shape.get_parameters(
                    key_prima_ub, layer_tag(self), prima_parameter_shape
                )
                abstract_shape.ub.coef -= self._multiply_prima_coefs_and_parameters(
                    prima_output_coefs, prima_ub_parameters
                )

        split_constraints = abstract_shape.get_split_constraints_for_relu(
            layer_tag(self), input_bounds
        )

        lb_intercept, ub_intercept = self._get_approximation_intercepts(
            input_bounds, split_constraints
        )

        new_lb_bias, new_ub_bias = abstract_shape._matmul_of_coef_and_interval(
            lb_intercept.unsqueeze(1), ub_intercept.unsqueeze(1)  # add query dimension
        )

        ub_slope = self._get_upper_approximation_slopes(
            config, input_bounds, split_constraints
        )

        # Backsub
        new_lb_form = self._backsub_affine_form(
            abstract_shape.lb,
            input_bounds,
            (new_lb_bias, new_ub_bias),
            ub_slope,
            prima_coefs,
            split_constraints,
            compute_upper_bound=False,
            abstract_shape=abstract_shape,
            config=config,
            prev_layer=prev_layer,
        )

        new_ub_form: Optional[AffineForm] = None
        if abstract_shape.ub is not None:
            new_ub_form = self._backsub_affine_form(
                abstract_shape.ub,
                input_bounds,
                (new_lb_bias, new_ub_bias),
                ub_slope,
                prima_coefs,
                split_constraints,
                compute_upper_bound=True,
                abstract_shape=abstract_shape,
                config=config,
                prev_layer=prev_layer,
            )

        abstract_shape.update_bounds(new_lb_form, new_ub_form)
        return abstract_shape

    def _backsub_affine_form(
        self,
        affine_form: AffineForm,
        input_bounds: Tuple[Tensor, Tensor],
        bias: Tuple[Tensor, Optional[Tensor]],
        ub_slope: Tensor,
        prima_coefs: Optional[Tuple[Tensor, Tensor, Tensor]],
        split_constraints: Optional[Tensor],
        compute_upper_bound: bool,
        abstract_shape: MN_BaB_Shape,
        config: BacksubstitutionConfig,
        prev_layer: Optional[AbstractModule],
    ) -> AffineForm:

        # Get parameters
        new_lb_bias, new_ub_bias = bias

        new_bias = new_ub_bias if compute_upper_bound else new_lb_bias

        lb_slope = self._get_lower_approximation_slopes(
            config,
            input_bounds,
            abstract_shape,
            key_alpha_relu(compute_upper_bound),
            split_constraints,
        )

        # Handle bias
        new_bias += affine_form.bias

        # Handle coef
        (
            new_lb_coef_tensor,
            new_ub_coef_tensor,
        ) = abstract_shape._elementwise_mul_of_coef_and_interval(lb_slope, ub_slope)

        new_coef_tensor = (
            new_ub_coef_tensor if compute_upper_bound else new_lb_coef_tensor
        )
        assert new_coef_tensor is not None

        # Handle prima contribution
        if prima_coefs is not None and prima_coefs[0].shape[2] > 0:

            (
                prima_output_coefs,
                prima_input_coefs,
                prima_const_coefs,
            ) = prima_coefs

            prima_parameter_shape = (
                abstract_shape.batch_size,
                prima_output_coefs.shape[2],
                1,
            )

            prima_parameters = abstract_shape.get_parameters(
                key_prima(compute_upper_bound), layer_tag(self), prima_parameter_shape
            )

            if compute_upper_bound:
                # sub prima const constraints to bias
                new_bias -= prima_const_coefs.bmm(prima_parameters).squeeze(-1)
                # sub prima input constraints to coefs
                new_coef_tensor -= self._multiply_prima_coefs_and_parameters(
                    prima_input_coefs, prima_parameters
                )
            else:
                # add prima input constraints to coefs
                new_bias += prima_const_coefs.bmm(prima_parameters).squeeze(-1)
                # add prima input constraints to coefs
                new_coef_tensor += self._multiply_prima_coefs_and_parameters(
                    prima_input_coefs, prima_parameters
                )

        # Handle split constraints
        if split_constraints is not None:
            # add betas, [B, 1, c, h, w]
            beta_contrib_shape = (abstract_shape.batch_size, 1, *self.output_dim)

            beta = abstract_shape.get_parameters(
                key_beta(compute_upper_bound), layer_tag(self), split_constraints.shape
            )
            beta_contrib = (beta * split_constraints).view(beta_contrib_shape)
            if compute_upper_bound:
                beta_contrib *= -1

            if abstract_shape.uses_dependence_sets():
                assert isinstance(affine_form.coef, DependenceSets)
                new_coef_tensor += DependenceSets.unfold_to(
                    beta_contrib, affine_form.coef
                )
            else:
                new_coef_tensor += beta_contrib

        # Create output
        new_coef: Union[Tensor, DependenceSets]

        if abstract_shape.uses_dependence_sets():
            assert isinstance(affine_form.coef, DependenceSets)
            new_coef = DependenceSets(
                new_coef_tensor,
                affine_form.coef.spatial_idxs,
                affine_form.coef.input_dim,
                affine_form.coef.cstride,
                affine_form.coef.cpadding,
            )
        else:
            new_coef = new_coef_tensor

        return AffineForm(new_coef, new_bias)

    def propagate_interval(
        self,
        interval: Tuple[Tensor, Tensor],
        use_existing_bounds: Optional[bool] = None,
        subproblem_state: Optional[SubproblemState] = None,
        activation_layer_only: bool = False,
        set_input: bool = True,
        set_output: bool = True,
    ) -> Tuple[Tensor, Tensor]:

        output_lb, output_ub = interval[0].clamp(min=0), interval[1].clamp(min=0)
        if (
            subproblem_state is not None
            and subproblem_state.constraints.split_state is not None
        ):
            subproblem_state.constraints.split_state.refine_split_constraints_for_relu(
                layer_tag(self), interval
            )
        # assert (output_ub >= output_lb).all()

        return output_lb, output_ub

    def propagate_abstract_element(
        self,
        abs_input: AbstractElement,
        use_existing_bounds: Optional[bool] = None,
        activation_layer_only: bool = False,
        set_input: bool = True,
        set_output: bool = True,
    ) -> AbstractElement:
        return abs_input.relu()[0]

    def _compute_new_prima_coefficients(
        self,
        prima_hyperparameters: PrimaHyperparameters,
        batch_size: int,
        intermediate_bounds_callback: Callable[[Tensor], Tuple[Tensor, Tensor]],
        device: torch.device,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        assert self.input_bounds

        # lb = self.input_bounds[0].detach().cpu()
        # ub = self.input_bounds[1].detach().cpu()
        # print(f"lb_avg_improve: {torch.mean(self.optim_input_bounds[0].reshape(self.input_bounds[0].shape) - self.input_bounds[0])} ub_avg_improve: {-1* torch.mean(self.optim_input_bounds[1].reshape(self.input_bounds[1].shape) - self.input_bounds[1])}")
        if self.optim_input_bounds is None:
            lb = self.input_bounds[0].detach().cpu()
            ub = self.input_bounds[1].detach().cpu()
        else:
            lb = (
                torch.max(
                    self.optim_input_bounds[0].reshape(self.input_bounds[0].shape),
                    self.input_bounds[0],
                )
                .detach()
                .cpu()
            )
            ub = (
                torch.min(
                    self.optim_input_bounds[1].reshape(self.input_bounds[1].shape),
                    self.input_bounds[1],
                )
                .detach()
                .cpu()
            )

        output_var_coefs, input_var_coefs, const_coefs = get_prima_constraints(
            lb,
            ub,
            ActivationType.ReLU,
            prima_hyperparameters,
            intermediate_bounds_callback,
            batch_size,
            self.output_dim,
        )

        n_prima_constraints = output_var_coefs.shape[2]
        assert output_var_coefs.shape == (
            batch_size,
            np.prod(self.output_dim),
            n_prima_constraints,
        )
        assert input_var_coefs.shape == (
            batch_size,
            np.prod(self.output_dim),
            n_prima_constraints,
        )
        assert const_coefs.shape == (
            batch_size,
            1,
            n_prima_constraints,
        )
        return (
            output_var_coefs.to(device),
            input_var_coefs.to(device),
            const_coefs.to(device),
        )

    def _get_prima_coefficients(  # TODO: move some of this logic into verification_subproblem
        self,
        config: BacksubstitutionConfig,
        abstract_shape: MN_BaB_Shape,
        intermediate_bounds_callback: Optional[
            Callable[[Tensor], Tuple[Tensor, Tensor]]
        ],
    ) -> Optional[Tuple[Tensor, Tensor, Tensor]]:
        if config.prima_hyperparameters is None:
            return None
        assert abstract_shape.subproblem_state is not None
        assert abstract_shape.subproblem_state.constraints.prima_constraints is not None
        prima_coefficients = (
            abstract_shape.subproblem_state.constraints.prima_constraints.prima_coefficients
        )
        if layer_tag(self) in prima_coefficients:
            return prima_coefficients[layer_tag(self)]
        if (
            intermediate_bounds_callback is None
        ):  # TODO: this seems like a bad way to configure this
            return None
        new_prima_coefficients = self._compute_new_prima_coefficients(
            config.prima_hyperparameters,
            abstract_shape.batch_size,
            intermediate_bounds_callback,
            abstract_shape.device,
        )
        prima_coefficients[layer_tag(self)] = new_prima_coefficients
        return new_prima_coefficients

    def _handle_reduced_parameter_sharing_for_lower_approximation_slopes(
        self,
        config: BacksubstitutionConfig,
        abstract_shape: MN_BaB_Shape,
    ) -> Tuple[
        Callable[
            [Tensor], Tensor
        ],  # lb_slope -> expanded_lb_slope (create appropriate number of copies of parameters)
        Callable[
            [Tensor], Tensor
        ],  # parameters -> selected parameters (select parameters corresponding to each active query)
    ]:
        # TODO: do not create parameters for stable neurons in the first place?
        if (
            config.parameter_sharing_config is None
            or layer_tag(self)
            not in config.layer_ids_for_which_to_reduce_parameter_sharing
        ):
            return lambda lb_slope: lb_slope, lambda params: params
        assert abstract_shape.subproblem_state is not None

        def filter_params(params: Tensor) -> Tensor:
            unstable_queries_in_starting_layer = (
                abstract_shape.get_unstable_queries_in_starting_layer()
            )
            if unstable_queries_in_starting_layer is not None:
                params = params[:, unstable_queries_in_starting_layer, :]
            return params

        for (
            layer_type,
            sharing_config,
        ) in config.parameter_sharing_config.entries:
            if abstract_shape.query_prev_layer is not None and is_layer_of_type(
                abstract_shape.query_prev_layer, layer_type
            ):
                if sharing_config == ParameterSharing.same_layer:

                    def keep_slope(lb_slope: Tensor) -> Tensor:
                        assert lb_slope.shape[1] == 1
                        return lb_slope  # default behavior, keep query dimension at 1 to share among all queries

                    def keep_params(params: Tensor) -> Tensor:
                        return params

                    return keep_slope, keep_params
                if sharing_config == ParameterSharing.none:

                    def expand_slope(lb_slope: Tensor) -> Tensor:
                        assert lb_slope.shape[1] == 1
                        repeats = abstract_shape.total_num_queries_in_starting_layer
                        return lb_slope.repeat(
                            1, repeats, *([1] * len(lb_slope.shape[2:]))
                        )

                    return expand_slope, filter_params
                if sharing_config == ParameterSharing.in_channel:
                    query_prev_layer_any = abstract_shape.query_prev_layer
                    assert isinstance(query_prev_layer_any, Conv2d)
                    query_prev_layer = query_prev_layer_any
                    num_channels = query_prev_layer.out_channels
                    num_queries = abstract_shape.total_num_queries_in_starting_layer
                    assert num_queries % num_channels == 0

                    def expand_slope(lb_slope: Tensor) -> Tensor:
                        assert lb_slope.shape[1] == 1
                        # create one set of parameters for each channel:
                        return lb_slope.repeat(
                            1, num_channels, *([1] * len(lb_slope.shape[2:]))
                        )

                    def select_params(params: Tensor) -> Tensor:
                        assert params.shape[0] == abstract_shape.batch_size
                        batch_size = abstract_shape.batch_size
                        # add dimension to expand:
                        resized = params.view(
                            batch_size, num_channels, 1, *params.shape[2:]
                        )
                        # share parameters within each channel: Note this does not allocate additional memory
                        replicated = resized.expand(
                            batch_size,
                            num_channels,
                            num_queries // num_channels,
                            *params.shape[2:],
                        )
                        # remove additional dimension:
                        params = replicated.reshape(
                            batch_size, num_queries, *params.shape[2:]
                        )
                        # filter out parameters corresponding to stable queries:
                        return filter_params(params)

                    return expand_slope, select_params
        # no config found for current layer type
        return lambda lb_slope: lb_slope, lambda params: params

    def _get_lower_approximation_slopes(
        self,
        config: BacksubstitutionConfig,
        bounds: Tuple[Tensor, Tensor],
        abstract_shape: Optional[MN_BaB_Shape] = None,
        parameter_key: Optional[ParameterTag] = None,
        split_constraints: Optional[Tensor] = None,
    ) -> Tensor:
        input_lb, input_ub = bounds

        lb_slope = torch.where(
            input_ub <= -input_lb,
            torch.zeros_like(input_lb),
            torch.ones_like(input_lb),
        ).unsqueeze(
            1
        )  # add query dimension

        if (
            abstract_shape is not None
            and abstract_shape.subproblem_state is not None
            and abstract_shape.subproblem_state.parameters.use_params
        ):
            assert parameter_key is not None

            (
                expand_slope,
                select_params,
            ) = self._handle_reduced_parameter_sharing_for_lower_approximation_slopes(
                config,
                abstract_shape,
            )

            def make_default(device: torch.device) -> Tensor:
                if config.relu_alpha_init_method == ReLUAlphaInitMethod.minimum_area:
                    default = lb_slope
                elif config.relu_alpha_init_method == ReLUAlphaInitMethod.one_half:
                    default = 0.5 * torch.ones_like(lb_slope)
                else:
                    raise RuntimeError("Unknown init method for ReLU alpha parameters")
                assert abstract_shape is not None
                default = expand_slope(default)

                return default.to(
                    device
                )  # TODO: it's created on 'cuda:0' and moved to 'cuda' here, why?

            lb_slope = abstract_shape.get_parameters(
                parameter_key,
                layer_tag(self),
                make_default_parameters=make_default,
            )

            lb_slope = self._set_slopes_of_stable_neurons(
                bounds, lb_slope, split_constraints
            )

            lb_slope = select_params(lb_slope)

            assert (
                lb_slope.shape[1] == 1
                or lb_slope.shape[1] == abstract_shape.num_queries
            ), "{} {} {}".format(
                lb_slope.shape, abstract_shape.query_id, abstract_shape.num_queries
            )
        else:
            lb_slope = self._set_slopes_of_stable_neurons(
                bounds, lb_slope, split_constraints
            )
        return lb_slope

    def _get_upper_approximation_slopes(
        self,
        config: BacksubstitutionConfig,
        bounds: Tuple[Tensor, Tensor],
        split_constraints: Optional[Tensor] = None,
    ) -> Tensor:
        input_lb, input_ub = bounds

        ub_slope = input_ub / (input_ub - input_lb + EPS)

        ub_slope = ub_slope.unsqueeze(1)  # add query dimension

        ub_slope = self._set_slopes_of_stable_neurons(
            bounds, ub_slope, split_constraints
        )

        return ub_slope

    def _set_slopes_of_stable_neurons(
        self,
        bounds: Tuple[Tensor, Tensor],
        slopes: Tensor,
        split_constraints: Optional[Tensor],
    ) -> Tensor:
        input_lb, input_ub = bounds

        inactive_relu_mask = (input_ub < 0).unsqueeze(1)
        active_relu_mask = (input_lb > 0).unsqueeze(1)

        if split_constraints is not None:
            inactive_relu_mask = inactive_relu_mask | (
                split_constraints == 1
            ).unsqueeze(1)
            active_relu_mask = active_relu_mask | (split_constraints == -1).unsqueeze(1)

        # slope of stable inactive ReLU is 0
        slopes = torch.where(inactive_relu_mask, torch.zeros_like(slopes), slopes)
        # slope of stable active ReLU is 1
        slopes = torch.where(active_relu_mask, torch.ones_like(slopes), slopes)

        return slopes

    def _get_approximation_intercepts(
        self,
        bounds: Tuple[Tensor, Tensor],
        split_constraints: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        input_lb, input_ub = bounds

        lb_intercept = torch.zeros_like(input_lb)
        ub_intercept = -input_lb * input_ub / (input_ub - input_lb + EPS)
        ub_intercept = self._set_intercepts_of_stable_neurons(
            bounds, ub_intercept, split_constraints
        )

        return lb_intercept, ub_intercept

    def _set_intercepts_of_stable_neurons(
        self,
        bounds: Tuple[Tensor, Tensor],
        ub_intercept: Tensor,
        split_constraints: Optional[Tensor],
    ) -> Tensor:
        input_lb, input_ub = bounds
        stable_node_mask = (input_ub < 0) | (input_lb > 0)

        if split_constraints is not None:
            stable_node_mask = stable_node_mask | (split_constraints != 0)

        return torch.where(
            stable_node_mask,
            torch.zeros_like(ub_intercept),
            ub_intercept,
        )

    def _multiply_prima_coefs_and_parameters(  # TODO: move this out
        self, prima_coefs: Tensor, prima_params: Tensor
    ) -> Tensor:
        batch_size = prima_coefs.shape[0]
        n_prima_constraints = prima_coefs.shape[2]
        assert prima_params.shape == (batch_size, n_prima_constraints, 1)

        temp = prima_coefs.bmm(prima_params)
        return temp.view(batch_size, 1, *self.output_dim)

    def get_activation_layer_ids(
        self, act_layer_ids: Optional[List[LayerTag]] = None
    ) -> List[LayerTag]:
        if act_layer_ids is None:
            act_layer_ids = []
        act_layer_ids.append(layer_tag(self))

        return act_layer_ids

    def get_relu_layer_ids(
        self, act_layer_ids: Optional[List[LayerTag]] = None
    ) -> List[LayerTag]:
        if act_layer_ids is None:
            act_layer_ids = []
        act_layer_ids.append(layer_tag(self))
        return act_layer_ids

    @classmethod
    def get_split_points(cls, lb: float, ub: float) -> float:
        return 0.0
