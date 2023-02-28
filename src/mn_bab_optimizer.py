from __future__ import annotations

import time
from typing import List, Optional, OrderedDict, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor, optim
from torch.optim import Optimizer

from src.abstract_domains.DP_f import DeepPoly_f
from src.abstract_domains.zonotope import HybridZonotope
from src.abstract_layers.abstract_network import AbstractNetwork
from src.exceptions.verification_timeout import VerificationTimeoutException
from src.mn_bab_shape import MN_BaB_Shape
from src.state.tags import LayerTag, query_tag
from src.utilities.config import (
    AbstractDomain,
    BacksubstitutionConfig,
    IntermediateBoundsMethod,
    MNBabOptimizerConfig,
)
from src.utilities.general import any_smaller
from src.verification_subproblem import SubproblemState, VerificationSubproblem


class MNBabOptimizer:
    def __init__(
        self,
        config: MNBabOptimizerConfig,
        backsubstitution_config: BacksubstitutionConfig,
    ) -> None:
        assert not (  # TODO: ensure this by freezing the config?
            config.prima.optimize and not config.alpha.optimize
        ), "If you optimize prima constraints, you also have to optimize alpha."

        self.config = config
        self.backsubstitution_config = backsubstitution_config

    def bound_root_subproblem(  # noqa: C901  # TODO: simplify function
        self,
        input_lb: Tensor,
        input_ub: Tensor,
        network: AbstractNetwork,
        query_coef: Tensor,
        early_stopping_threshold: Optional[float] = None,
        timeout: float = float("inf"),
        device: torch.device = torch.device("cpu"),
        initial_bounds: Optional[OrderedDict[LayerTag, Tuple[Tensor, Tensor]]] = None,
    ) -> Tuple[VerificationSubproblem, Optional[Tensor]]:

        backsubstitution_config = self.backsubstitution_config

        if self.config.parameter_sharing_config.reduce_parameter_sharing:
            layer_ids_for_which_to_reduce_parameter_sharing: Sequence[
                LayerTag
            ] = network.get_activation_layer_ids()

            if self.config.parameter_sharing_layer_id_filter is not None:
                layer_ids_for_which_to_reduce_parameter_sharing = (
                    self.config.parameter_sharing_layer_id_filter.filter_layer_ids(
                        layer_ids_for_which_to_reduce_parameter_sharing
                    )
                )

            backsubstitution_config = self.backsubstitution_config.with_parameter_sharing(
                parameter_sharing_config=self.config.parameter_sharing_config,
                layer_ids_for_which_to_reduce_parameter_sharing=layer_ids_for_which_to_reduce_parameter_sharing,
            )

        start_time = time.time()
        assert query_coef.shape[0] == 1, "Expected single query for root subproblem."

        query_coef = query_coef.to(device)
        network.reset_input_bounds()

        root_subproblem_state = SubproblemState.create_default(
            split_state=network.get_default_split_state(batch_size=1, device=device),
            optimize_prima=self.config.prima.optimize,
            batch_size=1,
            device=device,
            use_params=False,
        )

        if self.backsubstitution_config.domain_pass != AbstractDomain("none"):
            if self.backsubstitution_config.domain_pass == AbstractDomain("DPF"):
                input_abs_element: Union[
                    DeepPoly_f, HybridZonotope, None
                ] = DeepPoly_f.construct_from_bounds(
                    input_lb, input_ub, input_ub.dtype, domain="DPF"
                )
            elif self.backsubstitution_config.domain_pass in [
                AbstractDomain("zono"),
                AbstractDomain("box"),
                AbstractDomain("hbox"),
            ]:
                input_abs_element = HybridZonotope.construct_from_bounds(
                    input_lb,
                    input_ub,
                    input_ub.dtype,
                    domain=self.backsubstitution_config.domain_pass.value,
                )
            else:
                input_abs_element = None
            if input_abs_element is not None:
                with torch.no_grad():
                    network.set_layer_bounds_via_abstract_element_propagation(
                        input_abs_element,
                        activation_layer_only=True,
                        set_input=True,
                        set_output=False,
                    )
                root_subproblem_state.constraints.layer_bounds.improve(
                    network.get_current_intermediate_bounds()
                )

        root_subproblem_state.constraints.layer_bounds.improve(
            network.get_current_optimized_intermediate_bounds()
        )
        root_subproblem_state.constraints.update_split_constraints(
            network.get_relu_layer_ids(),
            root_subproblem_state.constraints.layer_bounds.intermediate_bounds,
        )

        # TODO @Mark This could be removed right now, correct?
        if initial_bounds is not None:
            root_subproblem_state.constraints.layer_bounds.improve(initial_bounds)

        (
            deep_poly_lbs,
            deep_poly_ubs,
            deep_poly_ub_inputs,
            _,
        ) = self.bound_minimum_with_deep_poly(
            backsubstitution_config,
            input_lb,
            input_ub,
            network,
            query_coef,
            subproblem_state=root_subproblem_state,
            reset_input_bounds=True,
            ibp_pass=self.backsubstitution_config.box_pass,
        )
        deep_poly_ub_inputs = deep_poly_ub_inputs.view(-1, *input_lb.shape[1:])
        root_subproblem_state.parameters.use_params = True
        assert not root_subproblem_state.constraints.is_infeasible.any()
        assert isinstance(deep_poly_lbs, Sequence)
        assert isinstance(deep_poly_ubs, Sequence)
        assert len(deep_poly_lbs) == len(deep_poly_lbs) == 1
        print("deep poly lower bounds:", deep_poly_lbs)
        if root_subproblem_state.constraints.split_state is not None:
            print(
                f"Unstable neurons post DP: {sum([(x==0).sum().item() for x in root_subproblem_state.constraints.split_state.split_constraints.values()])}"
            )

        # root node is never infeasible
        invalid_bounds_mask_root: Sequence[bool] = [False]
        if self._can_stop_early(
            deep_poly_lbs,
            deep_poly_ubs,
            early_stopping_threshold,
            invalid_bounds_mask_root,
        ):
            return (
                VerificationSubproblem.create_default(
                    deep_poly_lbs[0],
                    deep_poly_ubs[0],
                    split_state=None,
                    optimize_prima=False,
                    device=device,
                ),
                deep_poly_ub_inputs,
            )

        root_subproblem_state.constraints.layer_bounds.improve(
            network.get_current_intermediate_bounds()
        )

        if self.config.alpha.optimize:
            time_remaining = (start_time + timeout) - time.time()
            (alpha_lbs, alpha_ubs, alpha_ub_inputs) = self._bound_minimum_optimizing_alpha(
                backsubstitution_config,
                root_subproblem_state,  # updated in place
                input_lb,
                input_ub,
                network,
                query_coef,
                opt_iterations=self.config.alpha.opt_iterations,
                early_stopping_threshold=early_stopping_threshold,
                timeout=time_remaining,
                reset_input_bounds=True,
            )
            assert not root_subproblem_state.constraints.is_infeasible.any()
            assert len(alpha_lbs) == len(alpha_ubs) == 1
            print("alpha lower bounds:", alpha_lbs)
            if root_subproblem_state.constraints.split_state is not None:
                print(
                    f"Unstable neurons post alpha: {sum([(x == 0).sum().item() for x in root_subproblem_state.constraints.split_state.split_constraints.values()])}"
                )

            root_subproblem_state.constraints.layer_bounds.improve(
                network.get_current_intermediate_bounds()
            )

            if self._can_stop_early(
                alpha_lbs,
                alpha_ubs,
                early_stopping_threshold,
                invalid_bounds_mask_root,
            ):
                return (
                    VerificationSubproblem.create_default(
                        alpha_lbs[0],
                        alpha_ubs[0],
                        split_state=None,
                        optimize_prima=False,
                        device=device,
                    ),
                    alpha_ub_inputs,
                )

        if self.config.prima.optimize:
            time_remaining = (start_time + timeout) - time.time()
            (
                prima_lbs,
                prima_ubs,
                prima_ub_inputs,
            ) = self._bound_minimum_optimizing_alpha_prima(
                backsubstitution_config,
                root_subproblem_state,  # updated in place
                input_lb,
                input_ub,
                network,
                query_coef,
                opt_iterations=self.config.prima.opt_iterations,
                early_stopping_threshold=early_stopping_threshold,
                timeout=time_remaining,
                reset_input_bounds=True,
            )
            assert len(prima_lbs) == len(prima_ubs) == 1
            assert not root_subproblem_state.constraints.is_infeasible.any()
            print("prima lower bounds:", prima_lbs)
            if root_subproblem_state.constraints.split_state is not None:
                print(
                    f"Unstable neurons post prima: {sum([(x == 0).sum().item() for x in root_subproblem_state.constraints.split_state.split_constraints.values()])}"
                )
            if root_subproblem_state.constraints.prima_constraints is not None:
                print(
                    f"Total number of PRIMA constraints: {sum([x[0].shape[-1] for x in root_subproblem_state.constraints.prima_constraints.prima_coefficients.values()])}"
                )

            if self._can_stop_early(
                prima_lbs,
                prima_ubs,
                early_stopping_threshold,
                invalid_bounds_mask_root,
            ):
                return (
                    VerificationSubproblem.create_default(
                        prima_lbs[0],
                        prima_ubs[0],
                        split_state=None,
                        optimize_prima=False,
                        device=device,
                    ),
                    prima_ub_inputs,
                )

        best_lbs = deep_poly_lbs
        best_ubs = deep_poly_ubs
        if self.config.alpha.optimize:
            best_lbs = np.maximum(alpha_lbs, best_lbs).tolist()
            best_ubs = np.minimum(alpha_ubs, best_ubs).tolist()
        if self.config.prima.optimize:
            best_lbs = np.maximum(prima_lbs, best_lbs).tolist()
            best_ubs = np.minimum(prima_ubs, best_ubs).tolist()

        if backsubstitution_config.reduce_parameter_sharing:
            root_subproblem_state.parameters.modify_for_sharing()  # branch and bound expects a single set of parameters

        return (
            VerificationSubproblem(
                lower_bound=best_lbs[0],
                upper_bound=best_ubs[0],
                subproblem_state=root_subproblem_state,
                device=device,
            ),
            None,
        )

    def improve_subproblem_batch_bounds(
        self,
        subproblem_batch: SubproblemState,  # updated in place
        input_lb: Tensor,
        input_ub: Tensor,
        network: AbstractNetwork,
        query_coef: Tensor,
        early_stopping_threshold: Optional[float] = None,
        recompute_intermediate_bounds_after_branching: bool = True,
        timeout: float = float("inf"),
    ) -> Tuple[
        Sequence[float], Sequence[float], Tensor
    ]:  # TODO: add batch support for bounds directly into VerificationSubproblem
        assert self.config.alpha.optimize

        backsubstitution_config = self.backsubstitution_config

        if self.config.prima.optimize:
            (
                improved_lbs,
                improved_ubs,
                ub_inputs,
            ) = self._bound_minimum_optimizing_alpha_prima(
                backsubstitution_config,
                subproblem_batch,  # updated in place
                input_lb,
                input_ub,
                network,
                query_coef,
                self.config.prima.bab_opt_iterations,
                early_stopping_threshold,
                timeout,
                compute_upper_bound=recompute_intermediate_bounds_after_branching,  # Don't compute the upper_bound of the objective in case all intermediate bounds are fixed
                reset_input_bounds=True,
            )
        else:
            (
                improved_lbs,
                improved_ubs,
                ub_inputs,
            ) = self._bound_minimum_optimizing_alpha(
                backsubstitution_config,
                subproblem_batch,  # updated in place
                input_lb,
                input_ub,
                network,
                query_coef,
                self.config.alpha.bab_opt_iterations,
                early_stopping_threshold,
                timeout,
                compute_upper_bound=recompute_intermediate_bounds_after_branching,
                reset_input_bounds=True,
            )

        return (improved_lbs, improved_ubs, ub_inputs)

    def _can_stop_early(
        self,
        lower_bounds: Union[Sequence[float], Tensor],
        upper_bounds: Union[Sequence[float], Tensor],
        early_stopping_threshold: Optional[float],
        infeasibility_mask: Union[Sequence[bool], Tensor],
    ) -> bool:
        """
        Returns True if and only if a counterexample has been found
        or property is verified on all feasible subproblems.
        """
        if early_stopping_threshold is None:
            return all(infeasibility_mask)
        counter_example_found = any_smaller(upper_bounds, early_stopping_threshold)
        if counter_example_found:
            return True
        verified_mask = (
            lower_bound > early_stopping_threshold for lower_bound in lower_bounds
        )
        verified_or_infeasible = (
            verified or infeasible
            for (verified, infeasible) in zip(verified_mask, infeasibility_mask)
        )
        return all(verified_or_infeasible)

    @torch.no_grad()
    def bound_minimum_with_deep_poly(
        self,
        backsubstitution_config: BacksubstitutionConfig,
        input_lb: Tensor,
        input_ub: Tensor,
        network: AbstractNetwork,
        query_coef: Tensor,
        reset_input_bounds: bool,  # = True,
        return_tensors: bool = False,
        ibp_pass: bool = False,
        subproblem_state: Optional[SubproblemState] = None,
    ) -> Tuple[
        Union[Tensor, Sequence[float]], Union[Tensor, Sequence[float]], Tensor, Tensor
    ]:
        if (
            self.config.prima.optimize
            and subproblem_state is not None
            and subproblem_state.constraints.prima_constraints is not None
        ):
            backsubstitution_config = backsubstitution_config.with_prima(
                self.config.prima.hyperparameters, []
            )  # use existing prima constraints, but don't compute new ones

        if reset_input_bounds:
            network.reset_input_bounds()

        if ibp_pass:
            network.set_layer_bounds_via_interval_propagation(
                input_lb,
                input_ub,
                use_existing_bounds=True,
                subproblem_state=subproblem_state,
                activation_layer_only=True,
                has_batch_dim=True,
                set_input=True,
                set_output=False,
            )
            if subproblem_state is not None:
                subproblem_state.constraints.layer_bounds.improve(
                    network.get_current_intermediate_bounds()
                )
                subproblem_state.constraints.update_split_constraints(
                    network.get_relu_layer_ids(),
                    subproblem_state.constraints.layer_bounds.intermediate_bounds,
                )

        abstract_shape = network.get_mn_bab_shape(
            config=backsubstitution_config,
            input_lb=input_lb,
            input_ub=input_ub,
            query_id=query_tag(network),
            query_coef=query_coef,
            subproblem_state=subproblem_state,
            compute_upper_bound=False,
            reset_input_bounds=reset_input_bounds,
            recompute_intermediate_bounds=self.backsubstitution_config.intermediate_bounds_method
            >= IntermediateBoundsMethod["dp"],
            optimize_intermediate_bounds=False,
        )
        output_lbs, __ = abstract_shape.concretize(input_lb, input_ub)
        ubs_of_minimum, ub_inputs = MNBabOptimizer._get_upper_bound_of_minimum(
            input_lb,
            input_ub,
            network,
            query_coef,
            abstract_shape,
        )
        assert (
            (output_lbs <= ubs_of_minimum.to(output_lbs.device) + 1e-5)
            | (output_lbs.squeeze() >= 0)
        ).all(), f"output_lb: {output_lbs}; output_ub_min: {ubs_of_minimum}; output_ub_min: {ubs_of_minimum - output_lbs}"
        if return_tensors:
            return (
                output_lbs,  # type:ignore [return-value]
                ubs_of_minimum,
                ub_inputs,
                abstract_shape.lb.coef,
            )
        else:
            return (
                output_lbs.flatten().tolist(),  # type:ignore [return-value]
                ubs_of_minimum.flatten().tolist(),
                ub_inputs,
                abstract_shape.lb.coef,
            )

    def _bound_minimum_optimizing_alpha(
        self,
        backsubstitution_config: BacksubstitutionConfig,
        subproblem_state: SubproblemState,  # updated in place
        input_lb: Tensor,
        input_ub: Tensor,
        network: AbstractNetwork,
        query_coef: Tensor,
        opt_iterations: int,
        early_stopping_threshold: Optional[float] = None,
        timeout: float = float("inf"),
        compute_upper_bound: bool = True,
        reset_input_bounds: bool = True,
    ) -> Tuple[Sequence[float], Sequence[float], Tensor]:

        best_parameters = subproblem_state.parameters

        subproblem_state_without_prima = (
            subproblem_state.without_prima()
        )  # TODO: remove this

        (best_lbs, best_ubs, ub_inputs) = self._bound_minimum(
            backsubstitution_config=backsubstitution_config,
            input_lb=input_lb,
            input_ub=input_ub,
            network=network,
            query_coef=query_coef,
            subproblem_state=subproblem_state_without_prima,
            optimization_iterations=opt_iterations,
            early_stopping_threshold=early_stopping_threshold,
            timeout=timeout,
            compute_upper_bound=compute_upper_bound,
            reset_input_bounds=reset_input_bounds,
            recompute_intermediate_bounds=self.backsubstitution_config.intermediate_bounds_method
            >= IntermediateBoundsMethod["alpha"],
        )

        subproblem_state.parameters = (
            subproblem_state_without_prima.parameters
        )  # TODO: remove this

        assert subproblem_state.parameters == best_parameters

        return (best_lbs, best_ubs, ub_inputs)

    def _bound_minimum_optimizing_alpha_prima(
        self,
        backsubstitution_config: BacksubstitutionConfig,
        subproblem_state: SubproblemState,
        input_lb: Tensor,
        input_ub: Tensor,
        network: AbstractNetwork,
        query_coef: Tensor,
        opt_iterations: int,
        early_stopping_threshold: Optional[float] = None,
        timeout: float = float("inf"),
        compute_upper_bound: bool = True,  # whether or not to compute the UB portion of the backsub pass
        reset_input_bounds: bool = True,
    ) -> Tuple[Sequence[float], Sequence[float], Tensor]:

        assert (
            subproblem_state.constraints.prima_constraints is not None
        ), "prima constraints missing"

        layer_ids_for_which_to_compute_prima_constraints = (
            network.get_activation_layer_ids()
        )

        backsubstitution_config = backsubstitution_config.with_prima(
            self.config.prima.hyperparameters,
            layer_ids_for_which_to_compute_prima_constraints,
        )

        (best_lbs, best_ubs, ub_inputs) = self._bound_minimum(
            backsubstitution_config=backsubstitution_config,
            subproblem_state=subproblem_state,
            input_lb=input_lb,
            input_ub=input_ub,
            network=network,
            query_coef=query_coef,
            optimization_iterations=opt_iterations,
            early_stopping_threshold=early_stopping_threshold,
            timeout=timeout,
            compute_upper_bound=compute_upper_bound,
            reset_input_bounds=reset_input_bounds,
            recompute_intermediate_bounds=self.backsubstitution_config.intermediate_bounds_method
            >= IntermediateBoundsMethod["prima"],
        )

        return (best_lbs, best_ubs, ub_inputs)

    def _bound_minimum(
        self,
        backsubstitution_config: BacksubstitutionConfig,
        subproblem_state: SubproblemState,
        input_lb: Tensor,
        input_ub: Tensor,
        network: AbstractNetwork,
        query_coef: Tensor,
        optimization_iterations: int,
        early_stopping_threshold: Optional[float],
        timeout: float,
        compute_upper_bound: bool,
        reset_input_bounds: bool,
        recompute_intermediate_bounds: bool,
    ) -> Tuple[Sequence[float], Sequence[float], Tensor]:
        assert query_coef.is_leaf

        start_time = time.time()

        best_parameters = subproblem_state.parameters

        abstract_shape = network.get_mn_bab_shape(
            config=backsubstitution_config,
            input_lb=input_lb,
            input_ub=input_ub,
            query_id=query_tag(network),
            query_coef=query_coef,
            subproblem_state=subproblem_state,
            compute_upper_bound=compute_upper_bound,
            reset_input_bounds=reset_input_bounds,
            optimize_intermediate_bounds=False,
            recompute_intermediate_bounds=recompute_intermediate_bounds,
        )
        assert abstract_shape.subproblem_state is subproblem_state
        assert subproblem_state.parameters is best_parameters
        subproblem_state.parameters = (
            best_parameters.deep_copy()
        )  # (temporarily replace dict during optimization)

        (
            all_alpha_parameters,
            all_beta_parameters,
            all_prima_parameters,
            alpha_relu_parameters,
        ) = abstract_shape.get_optimizable_parameters(only_lb=not compute_upper_bound)

        parameters_to_optimize = [
            {"params": all_alpha_parameters, "lr": self.config.alpha.lr},
            {"params": all_beta_parameters, "lr": self.config.beta.lr},
            {"params": all_prima_parameters, "lr": self.config.prima.lr},
        ]

        optimizer = optim.Adam(parameters_to_optimize)
        scheduler = optim.lr_scheduler.OneCycleLR(  # type: ignore[attr-defined] # mypy bug?
            optimizer,
            self.config.max_lr(),
            optimization_iterations,
            final_div_factor=self.config.lr.final_div_factor,
        )

        best_lower_bounds = torch.tensor(
            [-float("inf")] * abstract_shape.batch_size, device=abstract_shape.device
        )
        best_upper_bounds = torch.tensor(
            [float("inf")] * abstract_shape.batch_size, device=abstract_shape.device
        )

        if (
            compute_upper_bound
        ):  # When compute upper bounds is false, we are not recomputing bounds
            abstract_shape.improve_layer_bounds(
                network.get_current_intermediate_bounds()
            )

        best_ub_inputs = torch.zeros_like(input_lb[0:1]).repeat(
            [query_coef.shape[0]] + (input_ub.dim() - 1) * [1]
        )
        assert query_coef.shape[1] == 1

        for i in range(optimization_iterations):
            if time.time() - start_time > timeout:
                raise VerificationTimeoutException(
                    lb=torch.min(best_lower_bounds).cpu().item(),
                    ub=torch.max(best_upper_bounds).cpu().item(),
                )
            abstract_shape = network.backsubstitute_mn_bab_shape(
                config=backsubstitution_config,
                input_lb=input_lb,
                input_ub=input_ub,
                query_coef=query_coef.detach(),
                abstract_shape=abstract_shape,
                compute_upper_bound=compute_upper_bound,
                reset_input_bounds=recompute_intermediate_bounds,
                optimize_intermediate_bounds=False,
                recompute_intermediate_bounds=recompute_intermediate_bounds,
            )
            output_lbs, __ = abstract_shape.concretize(input_lb, input_ub)
            assert abstract_shape.subproblem_state is subproblem_state

            # TODO we collapse query dim here. This can prevent simultaneous optimization for multiple output constraints
            output_lbs = output_lbs.flatten()
            upper_bounds, ub_inputs = MNBabOptimizer._get_upper_bound_of_minimum(
                input_lb,
                input_ub,
                network,
                query_coef,
                abstract_shape,
            )
            upper_bounds = upper_bounds.flatten()
            improvement_mask = output_lbs >= best_lower_bounds
            improvement_mask_ub = upper_bounds < best_upper_bounds
            best_parameters.improve(subproblem_state.parameters, improvement_mask)
            if (
                compute_upper_bound
            ):  # When compute upper bounds is false, we are not recomputing bounds
                abstract_shape.improve_layer_bounds(
                    network.get_current_intermediate_bounds()
                )

            best_ub_inputs = torch.where(
                improvement_mask_ub.view([-1] + (input_ub.dim() - 1) * [1]),
                ub_inputs.squeeze(1),
                best_ub_inputs,
            )

            best_lower_bounds = torch.maximum(output_lbs, best_lower_bounds).detach()
            best_upper_bounds = torch.minimum(upper_bounds, best_upper_bounds)

            if self._can_stop_early(
                best_lower_bounds,
                best_upper_bounds,
                early_stopping_threshold,
                abstract_shape.subproblem_state.is_infeasible,
            ):
                break

            not_yet_verified_lbs = (
                output_lbs[output_lbs < early_stopping_threshold]
                if early_stopping_threshold is not None
                else output_lbs
            )
            loss = -not_yet_verified_lbs.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if all(
                self._all_gradients_zero_mask(
                    optimizer, abstract_shape.batch_size, abstract_shape.device
                )
            ):
                break

            # Clamping only for ReLU
            for alpha_parameters in alpha_relu_parameters:
                alpha_parameters.data = torch.clamp(alpha_parameters.data, 0.0, 1.0)

            for beta_parameters in all_beta_parameters:
                beta_parameters.data = torch.clamp(beta_parameters.data, min=0.0)

            for prima_parameters in all_prima_parameters:
                prima_parameters.data = torch.clamp(prima_parameters.data, min=0.0)

        lb_list: Sequence[float] = best_lower_bounds.tolist()
        ub_list: Sequence[float] = best_upper_bounds.tolist()

        subproblem_state.parameters = best_parameters

        subproblem_state.update_feasibility()

        return (lb_list, ub_list, best_ub_inputs)

    @staticmethod
    def _get_upper_bound_of_minimum(
        input_lb: Tensor,
        input_ub: Tensor,
        network: AbstractNetwork,
        query_coef: Tensor,
        abstract_shape: MN_BaB_Shape,
    ) -> Tuple[Tensor, Tensor]:
        upper_bound_input = abstract_shape.get_input_corresponding_to_lower_bound(
            input_lb, input_ub
        )
        # Turns out torch is broken for large convolutions
        outputs: List[Tensor] = []
        for in_split in torch.split(
            upper_bound_input.view(-1, *input_lb.shape[1:]), 1000
        ):
            outputs.append(network(in_split))
        output = torch.cat(outputs)
        # output = network(upper_bound_input.view(-1, *input_lb.shape[1:]))

        return (
            torch.einsum("bij, bij -> bi", output.view_as(query_coef), query_coef),
            upper_bound_input,
        )

    def _all_gradients_zero_mask(
        self, optimizer: Optimizer, batch_size: int, device: torch.device
    ) -> Sequence[bool]:
        gradients_zero_mask = torch.tensor(
            [True for __ in range(batch_size)], device=device
        )
        for param_group in optimizer.param_groups:
            parameters = param_group["params"]
            for parameter_batch in parameters:
                if parameter_batch.grad is None:
                    continue
                assert parameter_batch.shape[0] == batch_size
                flattened_parameter_batch_grad = parameter_batch.grad.view(
                    parameter_batch.shape[0], -1
                )
                parameter_gradient_zero_mask = torch.all(
                    flattened_parameter_batch_grad == 0, dim=1
                )
                gradients_zero_mask = torch.logical_and(
                    gradients_zero_mask, parameter_gradient_zero_mask
                )
                if not gradients_zero_mask.any():
                    return gradients_zero_mask.tolist()
        return gradients_zero_mask.tolist()
