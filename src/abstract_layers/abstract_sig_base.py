from __future__ import annotations

from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from src.abstract_layers.abstract_module import AbstractModule
from src.mn_bab_shape import AffineForm, MN_BaB_Shape
from src.state.tags import (
    LayerTag,
    ParameterTag,
    key_alpha,
    key_beta,
    key_plus_lb,
    key_plus_ub,
    layer_tag,
)
from src.utilities.dependence_sets import DependenceSets
from src.verification_subproblem import SubproblemState


class SigBase(nn.Sigmoid, AbstractModule):
    def __init__(
        self,
        dim: Tuple[int, ...],
        act: Callable[[Tensor], Tensor],
        d_act: Callable[[Tensor], Tensor],
    ) -> None:
        super(SigBase, self).__init__()
        self.output_dim = dim
        self.dependence_set_block = False
        self.act = act
        self.d_act = d_act

    def update_input_bounds(
        self, input_bounds: Tuple[Tensor, Tensor], check_feasibility: bool = True
    ) -> None:
        input_bounds_shape_adjusted = (
            input_bounds[0].view(-1, *self.output_dim),
            input_bounds[1].view(-1, *self.output_dim),
        )
        super(SigBase, self).update_input_bounds(
            input_bounds_shape_adjusted, check_feasibility=check_feasibility
        )

    def _backsubstitute(
        self,
        abstract_shape: MN_BaB_Shape,
        tangent_points: Optional[Tensor],
        step_size: Optional[float],
        max_x: Optional[float],
        intermediate_bounds_callback: Optional[
            Callable[[Tensor], Tuple[Tensor, Tensor]]
        ] = None,
    ) -> MN_BaB_Shape:
        if self.input_bounds is None:
            raise RuntimeError("Cannot backsubstitute if bounds have not been set.")
        if tangent_points is None or step_size is None or max_x is None:
            raise RuntimeError(
                "Cannot compute Sig/Tanh bounds without pre-computed values"
            )

        (
            split_constraints,
            split_points,
        ) = abstract_shape.get_split_constraints_for_sig(
            layer_tag(self), self.input_bounds
        )

        # Backsub
        new_lb_form = self._backsub_affine_form(
            affine_form=abstract_shape.lb,
            input_bounds=self.input_bounds,
            tangent_points=tangent_points,
            step_size=step_size,
            max_x=max_x,
            prima_coefs=None,
            split_constraints=split_constraints,
            split_points=split_points,
            compute_upper_bound=False,
            abstract_shape=abstract_shape,
        )

        new_ub_form: Optional[AffineForm] = None
        if abstract_shape.ub is not None:
            new_ub_form = self._backsub_affine_form(
                affine_form=abstract_shape.ub,
                input_bounds=self.input_bounds,
                tangent_points=tangent_points,
                step_size=step_size,
                max_x=max_x,
                prima_coefs=None,
                split_constraints=split_constraints,
                split_points=split_points,
                compute_upper_bound=True,
                abstract_shape=abstract_shape,
            )

        abstract_shape.update_bounds(new_lb_form, new_ub_form)
        return abstract_shape

    def _backsub_affine_form(
        self,
        affine_form: AffineForm,
        input_bounds: Tuple[Tensor, Tensor],
        tangent_points: Tensor,
        step_size: float,
        max_x: float,
        prima_coefs: Optional[Tuple[Tensor, Tensor, Tensor]],
        split_constraints: Optional[Tensor],
        split_points: Optional[Tensor],
        compute_upper_bound: bool,
        abstract_shape: MN_BaB_Shape,
    ) -> AffineForm:

        # Get parameters

        (
            lb_slope,
            ub_slope,
            lb_intercept,
            ub_intercept,
        ) = SigBase._get_approximation_slopes_and_intercepts_for_act(
            input_bounds,
            tangent_points,
            step_size,
            max_x,
            self.act,
            self.d_act,
            abstract_shape,
            key_alpha(compute_upper_bound),
            layer_tag(self),
            split_constraints,
            split_points,
        )

        # Handle bias
        new_lb_bias, new_ub_bias = abstract_shape._matmul_of_coef_and_interval(
            lb_intercept.unsqueeze(1),  # add query dimension
            ub_intercept.unsqueeze(1),
        )
        new_bias = new_ub_bias if compute_upper_bound else new_lb_bias
        assert new_bias is not None

        new_bias += affine_form.bias

        # Handle coef
        new_coef: Optional[Union[Tensor, DependenceSets]]
        new_lb_coef, new_ub_coef = abstract_shape._elementwise_mul_of_coef_and_interval(
            lb_slope.unsqueeze(1), ub_slope.unsqueeze(1)  # add query dimension
        )
        new_coef = new_ub_coef if compute_upper_bound else new_lb_coef
        assert new_coef is not None

        # Handle Split constraints
        if split_constraints is not None:
            # add betas, [B, 1, c, h, w]
            #
            beta_contrib_shape = (abstract_shape.batch_size, 1, *self.output_dim)

            beta_lb = abstract_shape.get_parameters(
                key_beta(compute_upper_bound), layer_tag(self), split_constraints.shape
            )

            beta_contrib = (beta_lb * split_constraints).view(beta_contrib_shape)
            if compute_upper_bound:
                beta_contrib *= -1

            # Bias contribution
            beta_bias_shape = (abstract_shape.batch_size, *self.output_dim[1:])
            beta_bias_cont = beta_lb * split_constraints
            beta_bias_cont = (beta_bias_cont * split_points).sum(dim=1)
            if compute_upper_bound:
                beta_bias_cont *= -1

            new_bias -= beta_bias_cont.reshape(beta_bias_shape)

            # Coef contribution
            if abstract_shape.uses_dependence_sets():
                assert isinstance(affine_form.coef, DependenceSets)
                new_coef += DependenceSets.unfold_to(beta_contrib, affine_form.coef)
            else:
                new_coef += beta_contrib

        # Create output
        if abstract_shape.uses_dependence_sets():
            assert isinstance(affine_form.coef, DependenceSets)
            new_coef = DependenceSets(
                new_coef,
                affine_form.coef.spatial_idxs,
                affine_form.coef.input_dim,
                affine_form.coef.cstride,
                affine_form.coef.cpadding,
            )

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

        output_lb, output_ub = self.act(interval[0]), self.act(interval[1])
        assert (output_ub >= output_lb).all()

        return output_lb, output_ub

    @classmethod
    def _get_approximation_slopes_and_intercepts_for_act(
        cls,
        bounds: Tuple[Tensor, Tensor],
        tangent_points: Tensor,
        step_size: float,
        max_x: float,
        act: Callable[[Tensor], Tensor],
        d_act: Callable[[Tensor], Tensor],
        abstract_shape: Optional[MN_BaB_Shape] = None,
        parameter_key: Optional[ParameterTag] = None,
        layer_id: Optional[LayerTag] = None,
        split_constraints: Optional[Tensor] = None,
        split_points: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:

        input_lb, input_ub = bounds
        dtype = input_lb.dtype

        # Set bounds based on split constraints
        if split_constraints is not None:
            input_lb = torch.where(
                split_constraints == -1, torch.max(input_lb, split_points), input_lb  # type: ignore[arg-type] # mypy bug?
            )
            input_ub = torch.where(
                split_constraints == 1, torch.min(input_ub, split_points), input_ub  # type: ignore[arg-type] # mypy bug?
            )

        input_lb = torch.clamp(input_lb, min=-1 * (max_x - 1))
        input_ub = torch.clamp(input_ub, max=(max_x - 1))
        lb_convex_mask = input_lb >= 0
        ub_convex_mask = input_ub < 0

        # Compute the bounds on the tangent points
        idx = (
            torch.max(
                torch.zeros(input_lb.numel(), device=input_ub.device),
                (input_ub / step_size).to(torch.long).flatten(),
            )
            + 1
        ).long()
        lb_tangent_ubs = torch.index_select(tangent_points, 0, idx).view(input_lb.shape)

        idx = (
            torch.max(
                torch.zeros(input_ub.numel(), device=input_lb.device),
                (-1 * input_lb / step_size).to(torch.long).flatten(),
            )
            + 1
        ).long()
        ub_tangent_lbs = -1 * torch.index_select(tangent_points, 0, idx).view(
            input_ub.shape
        )

        lb_tangent_ubs = torch.min(lb_tangent_ubs, input_ub)
        ub_tangent_lbs = torch.max(ub_tangent_lbs, input_lb)

        if (
            abstract_shape is not None
            and abstract_shape.subproblem_state is not None
            and abstract_shape.subproblem_state.parameters.use_params
        ):
            assert parameter_key is not None
            assert layer_id is not None

            def make_default_lb(device: torch.device) -> Tensor:
                lb_init = ((lb_tangent_ubs + input_lb) / 2).detach()
                return lb_init.to(
                    device
                )  # TODO: it's created on 'cuda:0' and moved to 'cuda' here, why?

            def make_default_ub(device: torch.device) -> Tensor:
                ub_init = ((ub_tangent_lbs + input_ub) / 2).detach()
                return ub_init.to(
                    device
                )  # TODO: it's created on 'cuda:0' and moved to 'cuda' here, why?

            lb_tangent_points = abstract_shape.get_parameters(
                key_plus_lb(parameter_key), layer_id, make_default_lb
            )
            ub_tangent_points = abstract_shape.get_parameters(
                key_plus_ub(parameter_key), layer_id, make_default_ub
            )
        else:
            lb_tangent_points = (lb_tangent_ubs + input_lb) / 2
            ub_tangent_points = (ub_tangent_lbs + input_ub) / 2

        # if ub >= tangent_intersection_of_lb we can use the convex slope for our lower bound
        lb_convex_mask = (lb_convex_mask | (input_lb >= lb_tangent_ubs)).to(dtype)
        #
        ub_convex_mask = (ub_convex_mask | (input_ub <= ub_tangent_lbs)).to(dtype)

        # Note that these intervals may be empty, but only inf the second condition of the mask above holds and we are convex
        # Constrain lb_tangent_points to [lbs, min_lb_tangent]
        lb_tangent_points = torch.clamp(
            lb_tangent_points, min=input_lb, max=lb_tangent_ubs
        )
        # Constrain ub_tangent_points to [ub_tangent_lbs, ubs]
        ub_tangent_points = torch.clamp(
            ub_tangent_points, min=ub_tangent_lbs, max=input_ub
        )

        # Compute the slopes

        sigmoid_lb, sigmoid_ub = act(input_lb), act(input_ub)
        sigmoid_tlb, sigmoid_tub = act(lb_tangent_points), act(ub_tangent_points)
        # Convex slopes
        convex_slope = (sigmoid_ub - sigmoid_lb) / (input_ub - input_lb + 1e-6)
        convex_intercept = sigmoid_lb - input_lb * convex_slope
        # lb tangents
        tlb_slope = d_act(lb_tangent_points)
        tlb_intercept = sigmoid_tlb - lb_tangent_points * tlb_slope
        # ub tangents
        tub_slope = d_act(ub_tangent_points)
        tub_intercept = sigmoid_tub - ub_tangent_points * tub_slope

        # Final slopes and intercepts
        lb_slope = lb_convex_mask * convex_slope + (1 - lb_convex_mask) * tlb_slope
        lb_intercept = (
            lb_convex_mask * convex_intercept + (1 - lb_convex_mask) * tlb_intercept
        ) - 1e-6

        ub_slope = ub_convex_mask * convex_slope + (1 - ub_convex_mask) * tub_slope
        ub_intercept = (
            ub_convex_mask * convex_intercept + (1 - ub_convex_mask) * tub_intercept
        ) + 1e-6

        return lb_slope, ub_slope, lb_intercept, ub_intercept

    def get_activation_layer_ids(
        self, act_layer_ids: Optional[List[LayerTag]] = None
    ) -> List[LayerTag]:
        if act_layer_ids is None:
            act_layer_ids = []
        act_layer_ids.append(layer_tag(self))

        return act_layer_ids

    @classmethod
    def _compute_bound_to_tangent_point(
        cls, f: Callable[[Tensor], Tensor], d_f: Callable[[Tensor], Tensor]
    ) -> Tuple[Tensor, Tensor, float, float]:

        with torch.no_grad():
            max_x = 500
            step_size = 0.01
            num_points = int(max_x // step_size) + 1
            max_iter = 100

            def is_below(ip: Tensor, tp: Tensor) -> Tensor:
                """Return true if the tangent from tp intersects the function below ip

                Args:
                    ip (Tensor): intersection point
                    tp (Tensor): tangent point
                """
                return (d_f(tp) * (ip - tp) + f(tp) <= f(ip)).to(ip.dtype)

            ips = torch.linspace(0, max_x, num_points)  # Intersection points
            ub = torch.zeros_like(ips)  # Binary search upperbounds for each ip's tp
            lb = -1 * torch.ones_like(ips)

            # Adjust all lower bounds to be truely below
            while True:
                ib = is_below(ips, lb)
                lb = ib * lb + (1 - ib) * (2 * lb)
                if ib.sum() == ips.numel():
                    break

            for _ in range(max_iter):
                m = (lb + ub) / 2
                ib = is_below(ips, m)
                lb = ib * m + (1 - ib) * lb
                ub = ib * ub + (1 - ib) * m

            # By symmetry we have valid points negative ip as well
            return ips.clone(), lb.clone(), step_size, max_x
