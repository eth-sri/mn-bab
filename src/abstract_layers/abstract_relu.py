from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from src.abstract_layers.abstract_module import AbstractModule
from src.mn_bab_shape import MN_BaB_Shape
from src.utilities.dependence_sets import DependenceSets
from src.utilities.prima_interface import get_prima_constraints

EPS = 1e-15


class ReLU(nn.ReLU, AbstractModule):
    def __init__(self, dim: Tuple[int, ...]) -> None:
        super(ReLU, self).__init__()
        self.output_dim = dim
        self.dependence_set_block = False

    @classmethod
    def from_concrete_module(
        cls, module: nn.ReLU, input_dim: Tuple[int, ...], **kwargs: Any
    ) -> ReLU:
        return cls(input_dim)

    def update_input_bounds(self, input_bounds: Tuple[Tensor, Tensor]) -> None:
        input_bounds_shape_adjusted = (
            input_bounds[0].view(-1, *self.output_dim),
            input_bounds[1].view(-1, *self.output_dim),
        )
        super(ReLU, self).update_input_bounds(input_bounds_shape_adjusted)

    def backsubstitute(
        self,
        abstract_shape: MN_BaB_Shape,
        intermediate_bounds_callback: Optional[
            Callable[[Tensor], Tuple[Tensor, Tensor]]
        ] = None,
    ) -> MN_BaB_Shape:
        if self.input_bounds is None:
            raise RuntimeError("Cannot backsubstitute if bounds have not been set.")

        prima_coefs = self._get_prima_constraints(
            abstract_shape, intermediate_bounds_callback
        )
        prima_constraints_available = (
            prima_coefs is not None and prima_coefs[0].shape[2] > 0
        )
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
                "prima_lb", id(self), prima_parameter_shape
            )
            prima_ub_parameters = abstract_shape.get_parameters(
                "prima_ub", id(self), prima_parameter_shape
            )

        if prima_constraints_available:
            assert not abstract_shape.uses_dependence_sets()
            abstract_shape.lb_coef += self._mulitply_prima_coefs_and_parameters(
                prima_output_coefs, prima_lb_parameters
            )
            abstract_shape.ub_coef -= self._mulitply_prima_coefs_and_parameters(
                prima_output_coefs, prima_ub_parameters
            )

        if abstract_shape.split_constraints is not None:
            abstract_shape.refine_split_constraints_for(id(self), self.input_bounds)
            split_constraints = abstract_shape.split_constraints[id(self)]
        else:
            split_constraints = None
        lb_slope_for_lb = self._get_lower_approximation_slopes(
            self.input_bounds, abstract_shape, "alpha_lb", split_constraints
        )
        lb_slope_for_ub = self._get_lower_approximation_slopes(
            self.input_bounds, abstract_shape, "alpha_ub", split_constraints
        )
        ub_slope = self._get_upper_approximation_slopes(
            self.input_bounds, split_constraints
        )
        lb_intercept, ub_intercept = self._get_approximation_intercepts(
            self.input_bounds, split_constraints
        )

        new_lb_bias, new_ub_bias = abstract_shape._matmul_of_coef_and_interval(
            lb_intercept, ub_intercept
        )
        new_lb_bias += abstract_shape.lb_bias
        new_ub_bias += abstract_shape.ub_bias

        new_lb_coef, __ = abstract_shape._elementwise_mul_of_coef_and_interval(
            lb_slope_for_lb, ub_slope
        )
        __, new_ub_coef = abstract_shape._elementwise_mul_of_coef_and_interval(
            lb_slope_for_ub, ub_slope
        )

        if prima_constraints_available:
            assert not abstract_shape.uses_dependence_sets()
            # add prima const constraints to bias
            new_lb_bias += prima_const_coefs.bmm(prima_lb_parameters).squeeze(-1)
            new_ub_bias -= prima_const_coefs.bmm(prima_ub_parameters).squeeze(-1)

            # add prima input constraints to coefs
            new_lb_coef += self._mulitply_prima_coefs_and_parameters(
                prima_input_coefs, prima_lb_parameters
            )
            new_ub_coef -= self._mulitply_prima_coefs_and_parameters(
                prima_input_coefs, prima_ub_parameters
            )

        if split_constraints is not None:
            beta_lb = abstract_shape.get_parameters(
                "beta_lb", id(self), split_constraints.shape
            )
            beta_ub = abstract_shape.get_parameters(
                "beta_ub", id(self), split_constraints.shape
            )

            # add betas, [B, 1, c, h, w]
            beta_contrib_shape = (abstract_shape.batch_size, 1, *self.output_dim)
            lb_beta_contrib = (beta_lb * split_constraints).view(beta_contrib_shape)
            ub_beta_contrib = -(beta_ub * split_constraints).view(beta_contrib_shape)

            if abstract_shape.uses_dependence_sets():
                new_lb_coef += DependenceSets.unfold_to(
                    lb_beta_contrib, abstract_shape.lb_coef
                )
                new_ub_coef += DependenceSets.unfold_to(
                    ub_beta_contrib, abstract_shape.ub_coef
                )
            else:
                new_lb_coef += lb_beta_contrib
                new_ub_coef += ub_beta_contrib

        if abstract_shape.uses_dependence_sets():
            new_lb_coef = DependenceSets(
                new_lb_coef,
                abstract_shape.lb_coef.spatial_idxs,
                abstract_shape.lb_coef.cstride,
                abstract_shape.lb_coef.cpadding,
            )
            new_ub_coef = DependenceSets(
                new_ub_coef,
                abstract_shape.ub_coef.spatial_idxs,
                abstract_shape.ub_coef.cstride,
                abstract_shape.ub_coef.cpadding,
            )

        abstract_shape.update_bounds(new_lb_coef, new_ub_coef, new_lb_bias, new_ub_bias)
        return abstract_shape

    def propagate_interval(
        self, interval: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tensor]:

        output_lb, output_ub = interval[0].clamp(min=0), interval[1].clamp(min=0)
        # assert (output_ub >= output_lb).all()

        return output_lb, output_ub

    def _get_prima_constraints(
        self,
        abstract_shape: MN_BaB_Shape,
        intermediate_bounds_callback: Optional[
            Callable[[Tensor], Tuple[Tensor, Tensor]]
        ],
    ) -> Optional[Tuple[Tensor, Tensor, Tensor]]:
        if abstract_shape.uses_dependence_sets():
            return None
        if id(self) not in abstract_shape.prima_coefficients:
            if intermediate_bounds_callback is None:
                return None
            assert self.input_bounds
            assert abstract_shape.prima_hyperparameters

            output_var_coefs, input_var_coefs, const_coefs = get_prima_constraints(
                self.input_bounds[0].detach().cpu(),
                self.input_bounds[1].detach().cpu(),
                abstract_shape.prima_hyperparameters,
                intermediate_bounds_callback,
                abstract_shape.batch_size,
                self.output_dim,
            )

            n_prima_constraints = output_var_coefs.shape[2]
            assert output_var_coefs.shape == (
                abstract_shape.batch_size,
                np.prod(self.output_dim),
                n_prima_constraints,
            )
            assert input_var_coefs.shape == (
                abstract_shape.batch_size,
                np.prod(self.output_dim),
                n_prima_constraints,
            )
            assert const_coefs.shape == (
                abstract_shape.batch_size,
                1,
                n_prima_constraints,
            )
            abstract_shape.prima_coefficients[id(self)] = (
                output_var_coefs.to(abstract_shape.device),
                input_var_coefs.to(abstract_shape.device),
                const_coefs.to(abstract_shape.device),
            )

        return abstract_shape.prima_coefficients[id(self)]

    def _get_lower_approximation_slopes(
        self,
        bounds: Tuple[Tensor, Tensor],
        abstract_shape: Optional[MN_BaB_Shape] = None,
        parameter_key: Optional[str] = None,
        split_constraints: Optional[Tensor] = None,
    ) -> Tensor:
        input_lb, input_ub = bounds

        lb_slope = torch.where(
            input_ub <= -input_lb,
            torch.zeros_like(input_lb),
            torch.ones_like(input_lb),
        )
        if (
            abstract_shape is not None
            and abstract_shape.carried_over_optimizable_parameters is not None
        ):
            assert parameter_key is not None
            lb_slope = abstract_shape.get_parameters(
                parameter_key, id(self), lb_slope.shape, lb_slope
            )

        lb_slope = self._set_slopes_of_stable_neurons(
            bounds, lb_slope, split_constraints
        )
        return lb_slope

    def _get_upper_approximation_slopes(
        self, bounds: Tuple[Tensor, Tensor], split_constraints: Optional[Tensor] = None
    ) -> Tensor:
        input_lb, input_ub = bounds

        ub_slope = input_ub / (input_ub - input_lb + EPS)

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

        inactive_relu_mask = input_ub < 0
        active_relu_mask = input_lb > 0

        if split_constraints is not None:
            inactive_relu_mask = inactive_relu_mask | (split_constraints == 1)
            active_relu_mask = active_relu_mask | (split_constraints == -1)

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

    def _mulitply_prima_coefs_and_parameters(
        self, prima_coefs: Tensor, prima_params: Tensor
    ) -> Tensor:
        batch_size = prima_coefs.shape[0]
        n_prima_constraints = prima_coefs.shape[2]
        assert prima_params.shape == (batch_size, n_prima_constraints, 1)

        temp = prima_coefs.bmm(prima_params)
        return temp.view(batch_size, 1, *self.output_dim)

    def get_activation_layer_ids(
        self, act_layer_ids: Optional[List[int]] = None
    ) -> List[int]:
        if act_layer_ids is None:
            act_layer_ids = []
        act_layer_ids.append(id(self))

        return act_layer_ids
