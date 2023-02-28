# import itertools
import multiprocessing
import time
from typing import List, Optional, OrderedDict, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm  # type: ignore[import]

from src.abstract_domains.DP_f import DeepPoly_f
from src.abstract_domains.zonotope import HybridZonotope
from src.abstract_layers.abstract_network import AbstractNetwork
from src.branch_and_bound import BranchAndBound
from src.exceptions.verification_timeout import VerificationTimeoutException
from src.milp_network import MILPNetwork
from src.mn_bab_optimizer import MNBabOptimizer
from src.state.tags import LayerTag, layer_tag
from src.utilities.attacks import _evaluate_cstr, torch_whitebox_attack
from src.utilities.config import (
    AbstractDomain,
    DomainSplittingConfig,
    MNBabVerifierConfig,
)
from src.utilities.general import (
    batch_splits,
    compute_initial_splits,
    consolidate_input_regions,
    property_matrix_from_properties,
    split_input_regions,
    update_propertiy_matrices,
)
from src.utilities.output_property_form import OutputPropertyForm
from src.verification_subproblem import SubproblemState


class MNBaBVerifier:
    def __init__(
        self,
        network: AbstractNetwork,
        device: torch.device,
        config: MNBabVerifierConfig,
    ) -> None:
        self.network = network
        self.optimizer = MNBabOptimizer(config.optimizer, config.backsubstitution)
        self.bab = BranchAndBound(
            self.optimizer, config.bab, config.backsubstitution, device
        )
        self.config = config
        self.outer = config.outer
        self.domain_splitting = config.domain_splitting

        assert len(self.network.output_dim) == 1
        self.n_output_nodes = self.network.output_dim[0]

    def _verify_property_with_abstract_element(
        self,
        input_lb: Tensor,
        input_ub: Tensor,
        property_matrix: Tensor,
        early_stopping_thresholds: Tensor,
        abs_domain: AbstractDomain,
        compute_sensitivity: bool,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]:
        # print(
        #     "Verifying property_matrix\n",
        #     property_matrix,
        #     " against thresholds\n",
        #     early_stopping_thresholds,
        # )
        dtype = input_lb.dtype

        # if input_lb.dim() in [1, 3]:
        #     input_lb, input_ub = input_lb.unsqueeze(0), input_ub.unsqueeze(0)

        if abs_domain == AbstractDomain("DPF"):
            input_abs_element: Union[
                DeepPoly_f, HybridZonotope
            ] = DeepPoly_f.construct_from_bounds(
                input_lb, input_ub, dtype, domain="DPF"
            )
        elif abs_domain in [
            AbstractDomain("zono"),
            AbstractDomain("box"),
            AbstractDomain("hbox"),
        ]:
            if compute_sensitivity:
                center = (input_lb + input_ub) / 2.0
                width = ((input_ub - input_lb) / 2.0).requires_grad_(True)
                input_lb, input_ub = center - width, center + width
            input_abs_element = HybridZonotope.construct_from_bounds(
                input_lb, input_ub, dtype, domain=abs_domain.value
            )
        else:
            assert False, f"Unknown abstract domain {abs_domain}"

        if compute_sensitivity and abs_domain != AbstractDomain("DPF"):
            with torch.enable_grad():
                output_abs_element = (
                    self.network.set_layer_bounds_via_abstract_element_propagation(
                        input_abs_element,
                        use_existing_bounds=False,
                        set_input=False,
                        set_output=False,
                    )
                )
                (
                    verified,
                    falsified,
                    query_lb,
                    query_ub,
                    _,
                ) = output_abs_element.evaluate_queries(
                    property_matrix, early_stopping_thresholds
                )
                query_loss = (query_lb * (~verified)).sum()
                query_loss.backward()
                sensitivity = width.grad
        else:
            with torch.no_grad():
                output_abs_element = (
                    self.network.set_layer_bounds_via_abstract_element_propagation(
                        input_abs_element,
                        use_existing_bounds=False,
                        activation_layer_only=True,
                        set_input=False,
                        set_output=False,
                    )
                )
                (
                    verified,
                    falsified,
                    query_lb,
                    query_ub,
                    query_abs_element,
                ) = output_abs_element.evaluate_queries(
                    property_matrix, early_stopping_thresholds
                )
                sensitivity = None
                if abs_domain == AbstractDomain("DPF"):
                    assert isinstance(query_abs_element, DeepPoly_f)
                    assert query_abs_element.input_error_map is not None  # type: ignore # cant recognize that this is a DeepPoly_f

                    sensitivity = torch.zeros_like(input_lb).flatten(1)
                    sensitivity[
                        :, query_abs_element.input_error_map  # type: ignore # cant recognize that this is a DeepPoly_f
                    ] = query_abs_element.x_l_coef.flatten(  # type: ignore # cant recognize that this is a DeepPoly_f
                        2
                    ).abs().sum(
                        2
                    ) + query_abs_element.x_u_coef.flatten(  # type: ignore # cant recognize that this is a DeepPoly_f
                        2
                    ).abs().sum(
                        2
                    )
                    sensitivity = sensitivity.view(*input_lb.shape)
        return (verified, falsified, query_lb, query_ub, sensitivity)

    def _verify_output_form_with_deep_poly(
        self,
        input_lb: Tensor,
        input_ub: Tensor,
        output_form: OutputPropertyForm,
        compute_sensitivity: bool,
        subproblem_state: Optional[SubproblemState] = None,
        reset_input_bounds: bool = True,
        ibp_pass: bool = False,
    ) -> Tuple[OutputPropertyForm, Tensor, Tensor, Tensor, Tensor]:

        # Do a simple pass on all properties
        (
            deep_poly_lbs,
            ub_on_minimum,
            ub_inputs,
            sensitivity,
        ) = self.bab.optimizer.bound_minimum_with_deep_poly(
            self.bab.optimizer.backsubstitution_config,
            input_lb,
            input_ub,
            self.network,
            output_form.property_matrix,
            subproblem_state=subproblem_state,
            return_tensors=True,
            ibp_pass=ibp_pass,
            reset_input_bounds=reset_input_bounds,
        )
        bounds = deep_poly_lbs - output_form.property_threshold
        print(
            "deep poly lower bounds:",
            bounds.flatten().tolist(),
        )
        assert isinstance(deep_poly_lbs, Tensor)
        assert isinstance(ub_on_minimum, Tensor)
        verified: Tensor = deep_poly_lbs > output_form.property_threshold
        falsified: Tensor = ub_on_minimum < output_form.property_threshold

        # Update the output_form
        updated_out_form = output_form.update_properties_to_verify(
            verified,
            falsified,
            deep_poly_lbs,
            true_ub=False,
            easiest_and_first=False,
        )

        return (updated_out_form, bounds, verified, falsified, ub_inputs)

    def _verify_query_with_deep_poly(
        self,
        input_lb: Tensor,
        input_ub: Tensor,
        property_matrix: Tensor,
        early_stopping_thresholds: Tensor,
        compute_sensitivity: bool,
        subproblem_state: Optional[SubproblemState] = None,
        ibp_pass: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:

        # if input_lb.dim() in [1, 3]:
        #     input_lb, input_ub = input_lb.unsqueeze(0), input_ub.unsqueeze(0)

        (
            deep_poly_lbs,
            ub_on_minimum,
            ub_inputs,
            sensitivity,
        ) = self.bab.optimizer.bound_minimum_with_deep_poly(
            self.bab.optimizer.backsubstitution_config,
            input_lb,
            input_ub,
            self.network,
            property_matrix,
            subproblem_state=subproblem_state,
            return_tensors=True,
            ibp_pass=ibp_pass,
            reset_input_bounds=True,
        )
        assert isinstance(deep_poly_lbs, Tensor)
        assert isinstance(ub_on_minimum, Tensor)
        verified: Tensor = deep_poly_lbs > early_stopping_thresholds - 1e-6
        falsified: Tensor = ub_on_minimum < early_stopping_thresholds + 1e-6
        return (
            verified,
            falsified,
            deep_poly_lbs - early_stopping_thresholds,
            ub_on_minimum - early_stopping_thresholds,
            ub_inputs,
            sensitivity,
        )

    def _verify_property_with_deep_poly(
        self,
        input_lb: Tensor,
        input_ub: Tensor,
        label: int,
        competing_label: int,
        early_stopping_threshold: float,
    ) -> Tuple[bool, Tensor, Tensor]:
        with torch.no_grad():
            print("Verifying label ", label, " against ", competing_label)
            query_coef = torch.zeros(1, 1, self.n_output_nodes, device=input_lb.device)
            query_coef.data[0, 0, label] = 1
            query_coef.data[0, 0, competing_label] = -1

            subproblem_state = SubproblemState.create_default(
                split_state=None,
                optimize_prima=False,
                batch_size=1,
                device=input_lb.device,
                use_params=False,
            )

            (
                verified,
                falsified,
                deep_poly_lbs,
                deep_poly_ubs,
                ub_inputs,
                _,
            ) = self._verify_query_with_deep_poly(
                input_lb,
                input_ub,
                query_coef,
                early_stopping_threshold
                * torch.ones(
                    query_coef.shape[:-1],
                    device=query_coef.device,
                    dtype=query_coef.dtype,
                ),
                compute_sensitivity=False,
                subproblem_state=subproblem_state,
                ibp_pass=True,
            )
            print("deep poly lower bounds:", deep_poly_lbs[0])
            return (
                bool(verified[0]),
                deep_poly_lbs,
                deep_poly_ubs,
            )

    def _conduct_input_domain_splitting(
        self,
        domain_splitting: DomainSplittingConfig,
        queue: List[
            Tuple[
                Tensor,
                Tensor,
                Tuple[Tensor, Tensor, Tensor],
                int,
                Optional[Sequence[Sequence[Tuple[int, int, float]]]],
            ]
        ],
        time_out: float,
        properties_to_verify: Optional[Sequence[Sequence[Tuple[int, int, float]]]],
    ) -> Tuple[
        List[
            Tuple[
                Tensor,
                Optional[Tensor],
                Optional[Tuple[Tensor, Tensor, Tensor]],
                int,
                Optional[Sequence[Sequence[Tuple[int, int, float]]]],
            ]
        ],
        Tensor,
    ]:
        total_regions_considered = 0
        n_ver = 0

        out_queue: List[
            Tuple[
                Tensor,
                Tensor,
                Tuple[Tensor, Tensor, Tensor],
                int,
                Optional[Sequence[Sequence[Tuple[int, int, float]]]],
            ]
        ] = []

        query_lb_global = (
            torch.ones([1], dtype=queue[0][0].dtype, device=queue[0][0].device)
            * torch.inf
        )

        pbar = tqdm()
        while len(queue) > 0:
            if time.time() > time_out:
                print(
                    f"Input domain splitting timed out with queue length: {len(queue)}."
                )
                return list(queue + out_queue), query_lb_global
            length = len(queue)
            (
                input_lb,
                input_ub,
                property_matrix,
                property_threshold,
                combination_matrix,
                max_depth,
                properties_to_verify_batch,
            ) = batch_splits(queue, domain_splitting.batch_size)
            self.network.reset_input_bounds()
            self.network.reset_output_bounds()

            total_regions_considered += max_depth.shape[0]
            sensitivity: Optional[Tensor] = None
            if domain_splitting.domain == AbstractDomain("dp"):
                (
                    ids_verified,
                    ids_falsified,
                    query_lb,
                    query_ub,
                    ub_inputs,
                    sensitivity,
                ) = self._verify_query_with_deep_poly(
                    input_lb,
                    input_ub,
                    property_matrix,
                    property_threshold,
                    compute_sensitivity=True,
                    subproblem_state=SubproblemState.create_default(
                        split_state=None,
                        optimize_prima=False,
                        batch_size=input_lb.shape[0],
                        device=input_lb.device,
                        use_params=False,
                    ),
                    ibp_pass=True,
                )
            else:
                (
                    ids_verified,
                    ids_falsified,
                    query_lb,
                    query_ub,
                    sensitivity,
                ) = self._verify_property_with_abstract_element(
                    input_lb,
                    input_ub,
                    property_matrix,
                    property_threshold,
                    abs_domain=domain_splitting.domain,
                    compute_sensitivity=True,
                )
                ub_inputs = None
            (region_verified, region_falsified,) = update_propertiy_matrices(
                ids_verified,
                ids_falsified,
                property_matrix,
                property_threshold,
                combination_matrix,
                true_ub=domain_splitting.domain != AbstractDomain("dp"),
            )

            pbar.set_description(
                f"queue length: {length} | verified regions: {n_ver} | mean remaining depth: {max_depth.float().mean().cpu().item():.4f} | mean lb: {query_lb.mean():.5f} | mean ub: {query_ub.mean():.5f} | TR: {time_out - time.time():.2f}"
            )

            if properties_to_verify is not None:
                if ub_inputs is None:
                    adv_candidate = (input_lb + input_ub) / 2
                else:
                    adv_candidate = ub_inputs.view(-1, *input_lb.shape[1:])
                adv_found = ~_evaluate_cstr(
                    properties_to_verify,
                    self.network(adv_candidate).cpu().detach(),
                    torch_input=True,
                )
                if adv_found.any():
                    adv_idx = int(torch.tensor(adv_found).int().argmax().item())
                    adv_example = adv_candidate[adv_idx : adv_idx + 1]
                    return [
                        (adv_example, None, None, -1, properties_to_verify)
                    ], query_lb_global
            for batch_idx in range(ids_verified.shape[0]):
                if region_verified[batch_idx]:
                    n_ver += 1
                    if query_lb.shape[1] == 1:
                        query_lb_global = torch.minimum(
                            query_lb_global, query_lb[batch_idx]
                        )
                    # width = input_ub[batch_idx] - input_lb[batch_idx]
                    # corners = input_lb[batch_idx] + torch.tensor(list(itertools.product([0,1], repeat = width.shape[-1])), dtype=input_lb.dtype, device=input_lb.device) * width
                    # assert _evaluate_cstr(properties_to_verify, self.network(corners).cpu().detach(), torch_input=True).all()
                else:
                    if properties_to_verify_batch[batch_idx] is not None:
                        if ub_inputs is None:
                            adv_candidate = (
                                input_lb[batch_idx : batch_idx + 1]
                                + input_ub[batch_idx : batch_idx + 1]
                            ) / 2
                        else:
                            adv_candidate = ub_inputs[batch_idx].view(
                                -1, *input_lb.shape[1:]
                            )
                        adv_found = ~_evaluate_cstr(
                            properties_to_verify_batch[batch_idx],  # type: ignore # mypy can't parse condition
                            self.network(adv_candidate).cpu().detach(),
                            torch_input=True,
                        )
                        if adv_found.any():
                            return [
                                (
                                    adv_candidate,
                                    None,
                                    None,
                                    -1,
                                    properties_to_verify_batch[batch_idx],
                                )
                            ], query_lb_global
                    if max_depth[batch_idx] == 0:
                        out_queue += [
                            (
                                input_lb[batch_idx : batch_idx + 1],
                                input_ub[batch_idx : batch_idx + 1],
                                (
                                    property_matrix[batch_idx : batch_idx + 1],
                                    property_threshold[batch_idx : batch_idx + 1],
                                    combination_matrix[batch_idx : batch_idx + 1],
                                ),
                                0,
                                properties_to_verify_batch[batch_idx],
                            )
                        ]
                        continue
                    width = input_ub[batch_idx] - input_lb[batch_idx]
                    assert sensitivity is not None
                    split_dim = int(
                        ((sensitivity[batch_idx].abs().sum(0) + 1e-5) * width)
                        .argmax()
                        .item()
                    )
                    new_input_regions = split_input_regions(
                        [
                            (
                                input_lb[batch_idx : batch_idx + 1],
                                input_ub[batch_idx : batch_idx + 1],
                            )
                        ],
                        dim=split_dim,
                        splits=domain_splitting.split_factor,
                    )
                    queue = [
                        (
                            input_region[0],
                            input_region[1],
                            (
                                property_matrix[batch_idx : batch_idx + 1],
                                property_threshold[batch_idx : batch_idx + 1],
                                combination_matrix[batch_idx : batch_idx + 1],
                            ),
                            int(max_depth[batch_idx] - 1),
                            properties_to_verify_batch[batch_idx],
                        )
                        for input_region in new_input_regions
                    ] + queue

        print(f"A total of {total_regions_considered} regions considered.")
        return list(out_queue), query_lb_global

    def _verify_with_input_domain_splitting(
        self,
        domain_splitting: DomainSplittingConfig,
        input_lb: Tensor,
        input_ub: Tensor,
        properties_to_verify: List[List[Tuple[int, int, float]]],
        time_out: float,
    ) -> Tuple[
        List[
            Tuple[
                Tensor,
                Optional[Tensor],
                Optional[Tuple[Tensor, Tensor, Tensor]],
                int,
                Optional[Sequence[Sequence[Tuple[int, int, float]]]],
            ]
        ],
        Tensor,
    ]:
        num_classes = self.network.output_dim[-1]

        (
            property_matrix,
            property_threshold,
            combination_matrix,
        ) = property_matrix_from_properties(
            properties_to_verify, num_classes, input_lb.device, input_lb.dtype
        )

        queue: List[
            Tuple[
                Tensor,
                Tensor,
                Tuple[Tensor, Tensor, Tensor],
                int,
                Optional[Sequence[Sequence[Tuple[int, int, float]]]],
            ]
        ] = compute_initial_splits(
            input_lb,
            input_ub,
            property_matrix,
            property_threshold,
            combination_matrix,
            domain_splitting,
        )
        return self._conduct_input_domain_splitting(
            domain_splitting, queue, time_out, properties_to_verify
        )

    # TODO Remove
    def _verify_output_form_with_bab(
        self,
        output_form: OutputPropertyForm,
        input_lb: Tensor,
        input_ub: Tensor,
        sample_id: int,
        properties_to_verify_orig: List[List[Tuple[int, int, float]]],
        timeout: float,
    ) -> Tuple[
        bool,
        Optional[List[np.ndarray]],
        Optional[List[Tuple[int, int, float]]],
        Optional[float],
        Optional[float],
    ]:
        unit_clauses = output_form.properties_to_verify

        for const_list in unit_clauses:
            # TODO add sorting by dp bounds
            is_verified = False
            pot_ub_inputs: Optional[Tensor] = None
            for gt_tuple in const_list:
                if is_verified:
                    break
                if time.time() > timeout:
                    print("Verification timed out.")
                    return (False, None, const_list, -float("inf"), float("inf"))

                property_id = f"sample_{str(sample_id)}_gt_tuple_{gt_tuple[0]}_{gt_tuple[1]}_{gt_tuple[2]}"
                time_remaining = timeout - time.time()
                is_verified, best_lb, best_ub, pot_ub_inputs = self.verify_property(
                    property_id,
                    input_lb,
                    input_ub,
                    gt_tuple,
                    time_remaining,
                )
                if pot_ub_inputs is not None:
                    adv_found = ~_evaluate_cstr(
                        properties_to_verify_orig,
                        self.network(pot_ub_inputs).detach().cpu(),
                        torch_input=True,
                    )
                    if adv_found.any():
                        print("Adex found via BaB")
                        bab_adv_example = pot_ub_inputs[
                            int(torch.tensor(adv_found).int().argmax().item())
                        ].unsqueeze(0)
                        return (
                            False,
                            [np.array(bab_adv_example.cpu())],
                            const_list,
                            None,
                            None,
                        )
            if not is_verified:
                return (
                    False,
                    None,
                    const_list,
                    best_lb,
                    best_ub,
                )  # TODO: reconstruct adversarial example?

        return (True, None, None, 0, float("inf"))

    # TODO Remove
    def verify(
        self,
        sample_id: int,
        input: Tensor,
        input_lb: Tensor,
        input_ub: Tensor,
        label: int,
        timeout: float,
        num_classes: int = 10,
    ) -> Tuple[bool, bool]:  # (verified, disproved)
        start_time = time.time()

        def generate_constraints(
            class_num: int, y: int
        ) -> Sequence[Sequence[Tuple[int, int, float]]]:
            # Constraints is an "and" list of "or" clauses each of the format y[j]-y[i] > c with (i,j,c)
            return [[(y, i, 0.0)] for i in range(class_num) if i != y]

        properties_to_verify = []
        for constraint_list in generate_constraints(num_classes, int(label)):
            true_label, competing_label, early_stopping_threshold = constraint_list[0]
            # TODO correctly handle list of "or" properties
            (dp_verified, dp_lbs, dp_ubs) = self._verify_property_with_deep_poly(
                input_lb,
                input_ub,
                true_label,
                competing_label,
                early_stopping_threshold,
            )
            if not dp_verified:
                properties_to_verify.append(constraint_list)

        if not properties_to_verify:
            # All properties already verified
            return True, False

        adversarial_example, __ = torch_whitebox_attack(
            self.network,
            input_lb.device,
            input,
            properties_to_verify,
            input_lb,
            input_ub,
            restarts=5,
        )
        if adversarial_example is not None:
            return False, True

        for constraint_list in properties_to_verify:
            verified = False
            for gt_tuple in constraint_list:
                if verified:
                    break

                if time.time() - start_time > timeout:
                    print("Verification timed out.")
                    return False, False

                property_id = f"sample_{str(sample_id)}_gt_tuple_{gt_tuple[0]}_{gt_tuple[1]}_{gt_tuple[2]}"

                time_remaining = timeout - (time.time() - start_time)
                verified, lb, ub, ub_inputs = self.verify_property(
                    property_id,
                    input_lb,
                    input_ub,
                    gt_tuple,
                    time_remaining,
                )
                if ub_inputs is not None:
                    adv_found = ~_evaluate_cstr(
                        properties_to_verify,
                        self.network(ub_inputs).detach().cpu(),
                        torch_input=True,
                    )
                    if adv_found.any():
                        print("Adex found via BaB")
                        # adv_bab_example = ub_inputs[
                        #     int(torch.tensor(adv_found).int().argmax().item())
                        # ].unsqueeze(0)
                        return False, True

        return verified, False

    def verify_property(
        self,
        property_id: str,
        input_lb: Tensor,
        input_ub: Tensor,
        gt_tuple: Tuple[int, int, float],
        timeout: float,
        initial_bounds: Optional[OrderedDict[LayerTag, Tuple[Tensor, Tensor]]] = None,
    ) -> Tuple[bool, float, float, Optional[Tensor]]:
        print("Verifying gt tuple ", gt_tuple)
        query_coef = torch.zeros(1, 1, self.n_output_nodes)
        early_stopping_threshold = gt_tuple[2] * (-1 if gt_tuple[0] == -1 else 1)
        if gt_tuple[0] != -1:
            query_coef[0, 0, gt_tuple[0]] = 1
        if gt_tuple[1] != -1:
            query_coef[0, 0, gt_tuple[1]] = -1

        return self.verify_query(
            property_id,
            input_lb,
            input_ub,
            query_coef,
            early_stopping_threshold,
            timeout,
            adapter=None,
            initial_bounds=initial_bounds,
        )

    def verify_query(
        self,
        property_id: str,
        input_lb: Tensor,
        input_ub: Tensor,
        query_coef: Tensor,
        early_stopping_threshold: float,
        timeout: float,
        adapter: Optional[nn.Sequential] = None,
        initial_bounds: Optional[OrderedDict[LayerTag, Tuple[Tensor, Tensor]]] = None,
    ) -> Tuple[bool, float, float, Optional[Tensor]]:
        print("Verifying query ", property_id)

        try:
            if self.network.has_output_adapter:
                self.bab.config.batch_sizes = tuple(  # type:ignore [assignment]
                    list(self.bab.config.batch_sizes)
                    + [self.bab.config.batch_sizes[-1]]
                )

            (
                property_lb,
                property_ub,
                ub_inputs,
            ) = self.bab.bound_minimum_with_branch_and_bound(
                property_id,
                query_coef,
                self.network,
                input_lb,
                input_ub,
                early_stopping_threshold=early_stopping_threshold,
                timeout=timeout,
                initial_bounds=initial_bounds,
            )

            if self.network.has_output_adapter:
                self.bab.config.batch_sizes = self.bab.config.batch_sizes[:-1]

            return (
                property_lb >= early_stopping_threshold,
                property_lb,
                property_ub,
                ub_inputs,
            )
        except VerificationTimeoutException as e:
            print(f"Verification of property {property_id} timed out")
            lb = e.best_lb
            ub = e.best_ub
            assert lb is not None and ub is not None
            return False, lb, ub, None

    def verify_via_config(  # noqa C901 # ignore too complex function
        self,
        sample_id: int,
        input: Tensor,
        input_lb: Tensor,
        input_ub: Tensor,
        output_form: OutputPropertyForm,
        # properties_to_verify: List[List[Tuple[int, int, float]]],
        timeout: float = 400.0 + time.time(),
    ) -> Tuple[
        bool,
        Optional[List[np.ndarray]],
        Optional[List[Tuple[int, int, float]]],
        Optional[float],
        Optional[float],
    ]:

        if not output_form.properties_to_verify:
            return True, None, None, None, None

        properties_to_verify_orig = output_form.properties_to_verify.copy()

        self.network.reset_input_bounds()
        self.network.reset_optim_input_bounds()

        if self.outer.forward_dp_pass:
            with torch.no_grad():
                self.network.set_layer_bounds_via_forward_dp_pass(
                    self.config.backsubstitution,
                    input_lb=input_lb,
                    input_ub=input_ub,
                    timeout=0.8 * (timeout - time.time()) + time.time(),
                )
                self.network.activation_layer_bounds_to_optim_layer_bounds()

        if self.outer.initial_dp:

            (
                dp_out_prop_form,
                bounds,
                verified,
                falsified,
                ub_inputs,
            ) = self._verify_output_form_with_deep_poly(
                input_lb,
                input_ub,
                output_form,
                compute_sensitivity=False,
                subproblem_state=SubproblemState.create_default(
                    split_state=self.network.get_default_split_state(
                        batch_size=input_lb.shape[0], device=input_lb.device
                    ),
                    optimize_prima=False,
                    batch_size=1,
                    device=input_lb.device,
                    use_params=False,
                ),
                reset_input_bounds=not self.outer.forward_dp_pass,
                ibp_pass=True,
            )
            if falsified.any():
                property_falsified = ~_evaluate_cstr(
                    properties_to_verify_orig,
                    self.network(ub_inputs[falsified]).cpu().detach().numpy(),
                )
                if property_falsified.any():
                    print("Adex found via initial DP")
                    adv_example = ub_inputs[falsified][
                        int(torch.tensor(property_falsified).int().argmax())
                    ].view_as(input_lb)
                    return (False, [np.array(adv_example.cpu())], None, None, None)

            output_form = dp_out_prop_form
        properties_to_verify = output_form.properties_to_verify

        if not properties_to_verify:
            return (True, None, None, 0, 0)

        if self.outer.adversarial_attack:
            adversarial_example, __ = torch_whitebox_attack(
                self.network,
                input_lb.device,
                input,
                properties_to_verify,
                input_lb,
                input_ub,
                restarts=self.outer.adversarial_attack_restarts,
            )
            if adversarial_example is not None:
                selected_adv_example = (
                    torch.tensor(adversarial_example[0])
                    .unsqueeze(0)
                    .to(input_lb.device)
                )
                assert ~_evaluate_cstr(
                    properties_to_verify_orig,
                    self.network(selected_adv_example).cpu().detach().numpy(),
                )
                return (False, [selected_adv_example], None, None, None)

        if (
            self.outer.milp_config.refine_via_milp > 0
            and not self.outer.milp_config.solve_via_milp
        ):

            milp_cfg = self.outer.milp_config
            if milp_cfg.pre_refine_via_ab_prima:
                # Dummy root subproblem bounding to reduce #unstable_neurons
                gt_tuple = properties_to_verify[0][0]
                query_coef = torch.zeros(1, 1, self.n_output_nodes)
                early_stopping_threshold = gt_tuple[2]
                if gt_tuple[0] != -1:
                    query_coef[0, 0, gt_tuple[0]] = 1
                if gt_tuple[1] != -1:
                    query_coef[0, 0, gt_tuple[1]] = -1

                _, adex_candidate = self.optimizer.bound_root_subproblem(
                    input_lb,
                    input_ub,
                    self.network,
                    query_coef,
                    early_stopping_threshold,
                    timeout,
                    input_lb.device,
                )

                if adex_candidate is not None and ~_evaluate_cstr(
                    properties_to_verify_orig,
                    self.network(adex_candidate.to(device=input_lb.device))
                    .cpu()
                    .detach()
                    .numpy(),
                ):
                    print("Adex found via MILP refine")
                    return (
                        False,
                        [np.array(adex_candidate.cpu())],
                        None,
                        None,
                        None,
                    )

            milp_timeout = (
                min(time.time() + milp_cfg.timeout_refine_total, timeout)
                if milp_cfg.timeout_refine_total
                else timeout
            )

            milp_instance_timeout = (
                milp_cfg.timeout_refine_neuron
                if milp_cfg.timeout_refine_neuron
                else milp_timeout
            )

            for i, act_layer_id in enumerate(self.network.get_activation_layer_ids()):
                if i >= milp_cfg.refine_via_milp:
                    break
                if i == 0:
                    continue
                layer = self.network.layer_id_to_layer[act_layer_id]
                model = (
                    MILPNetwork.build_model_from_abstract_net(  # Build restricted model
                        (input_lb + input_ub) / 2,
                        input_lb,
                        input_ub,
                        self.network,
                        up_to_layer_id=act_layer_id,
                    )
                )

                layer.optim_input_bounds = model.get_network_bounds_at_layer_multi(
                    layer_tag(layer),
                    compute_input_bounds=True,
                    timeout_per_instance=milp_instance_timeout,
                    timeout=milp_timeout,
                    NUMPROCESSES=multiprocessing.cpu_count(),
                    refine_only_unstable=milp_cfg.refine_only_unstable,
                )
                assert layer.optim_input_bounds is not None
                assert layer.input_bounds is not None
                layer.optim_input_bounds = (
                    layer.optim_input_bounds[0]
                    .view_as(layer.input_bounds[0])
                    .to(input_lb.device),
                    layer.optim_input_bounds[1]
                    .view_as(layer.input_bounds[1])
                    .to(input_lb.device),
                )
                if layer.input_bounds is not None:
                    unstable_before = (
                        (layer.input_bounds[0] * layer.input_bounds[1] < 0)
                        .float()
                        .sum()
                    )
                    unstable_after = (
                        (layer.optim_input_bounds[0] * layer.optim_input_bounds[1] < 0)
                        .float()
                        .sum()
                    )
                    print(f"Before: {unstable_before} After: {unstable_after}")

        if self.outer.milp_config.solve_via_milp:
            batched_input_lb = input_lb
            batched_input_ub = input_ub

            model = MILPNetwork.build_model_from_abstract_net(  # Build restricted model
                (batched_input_lb + batched_input_ub) / 2,
                batched_input_lb,
                batched_input_ub,
                self.network,
            )
            is_verified, _, counter_example = model.verify_properties(
                properties_to_verify,
                timeout_per_instance=timeout,
                timeout_total=timeout,
                start_time=time.time(),
                NUMPROCESSES=multiprocessing.cpu_count(),
            )
            if not is_verified:
                if counter_example is not None:
                    counter_example = counter_example[: input_lb.numel()].view_as(
                        input_lb
                    )
                    assert ~_evaluate_cstr(
                        properties_to_verify_orig,
                        self.network(counter_example.to(device=input_lb.device))
                        .cpu()
                        .detach()
                        .numpy(),
                    )
                    print("Adex found via MILP")
                    return (
                        is_verified,
                        [np.array(counter_example.cpu())],
                        None,
                        None,
                        None,
                    )

            return is_verified, None, None, None, None

        if self.outer.input_domain_splitting:
            queue, _ = self._verify_with_input_domain_splitting(
                self.domain_splitting,
                input_lb.clone(),
                input_ub.clone(),
                properties_to_verify,
                timeout,
            )
            if len(queue) == 0:
                return (True, None, None, 0, 0)
            elif len(queue) == 1 and queue[0][-2] == -1:
                # counterexample region returned
                adv_example = queue[0][0]
                out = self.network(adv_example)
                assert (
                    (input_lb <= adv_example + 1e-8)
                    .__and__(input_ub >= adv_example - 1e-8)
                    .all()
                )
                if not _evaluate_cstr(
                    properties_to_verify_orig, out.detach(), torch_input=True
                ):
                    print("Adex found via splitting")
                    return (False, [np.array(adv_example.cpu())], None, None, None)
                else:
                    assert False, "should have been a counterexample"
            else:
                has_no_x1 = None in [x[1] for x in queue]
                assert not has_no_x1, "x[1] not set for value in queue"
                input_lb, input_ub = consolidate_input_regions(
                    [(x[0], x[1]) for x in queue if x[1] is not None]
                )
                self.network.reset_input_bounds()

        if time.time() > timeout:
            return False, None, None, None, None

        return self._verify_output_form_with_bab(
            output_form,
            input_lb,
            input_ub,
            sample_id,
            properties_to_verify_orig,
            timeout,
        )

    def verify_unet_via_config(  # noqa C901 # ignore too complex function
        self,
        sample_id: int,
        input: Tensor,
        input_lb: Tensor,
        input_ub: Tensor,
        output_form: OutputPropertyForm,
        verification_target: int,
        # properties_to_verify: List[List[Tuple[int, int, float]]],
        timeout: float = 400.0 + time.time(),
    ) -> Tuple[
        bool,
        Optional[List[np.ndarray]],
        Optional[List[Tuple[int, int, float]]],
        Optional[float],
        Optional[float],
    ]:
        verified_so_far = 0
        given_up_on_so_far = 0
        properties_to_verify_orig = output_form.properties_to_verify.copy()
        remaining = len(properties_to_verify_orig)

        self.network.reset_input_bounds()

        if self.outer.forward_dp_pass:
            with torch.no_grad():
                self.network.set_layer_bounds_via_forward_dp_pass(
                    self.config.backsubstitution,
                    input_lb=input_lb,
                    input_ub=input_ub,
                    timeout=0.8 * (timeout - time.time()) + time.time(),
                )
                self.network.activation_layer_bounds_to_optim_layer_bounds()

        if self.outer.initial_dp:

            sub_state = SubproblemState.create_default(
                split_state=None,
                optimize_prima=False,
                batch_size=1,
                device=input_lb.device,
                use_params=False,
            )
            sub_state.constraints.layer_bounds.improve(
                self.network.get_current_intermediate_bounds()
            )

            (
                dp_out_prop_form,
                bounds,
                verified,
                falsified,
                ub_inputs,
            ) = self._verify_output_form_with_deep_poly(
                input_lb,
                input_ub,
                output_form,
                compute_sensitivity=False,
                subproblem_state=sub_state,
                reset_input_bounds=True,
                ibp_pass=True,
            )

            verified_so_far += int(verified.float().sum().item())
            given_up_on_so_far += int(falsified.float().sum().item())
            remaining -= verified_so_far + given_up_on_so_far

            if (
                given_up_on_so_far
                >= len(properties_to_verify_orig) - verification_target
            ):
                incorrect_pixels = (
                    self.network(ub_inputs.view(-1, *input.shape[1:])[:, :3])
                    .view(-1, 2, *input.shape[2:])
                    .argmax(1)
                    != input[:, 3]
                ).sum((1, 2))
                max_incorrect_pixels, adv_id = incorrect_pixels.max(0)
                if (
                    max_incorrect_pixels
                    > len(properties_to_verify_orig) - verification_target
                ):
                    print(f"Found adv example for UNET")
                    return (
                        False,
                        [
                            np.array(
                                ub_inputs.view(-1, *input.shape[1:])[
                                    adv_id : adv_id + 1
                                ].cpu()
                            )
                        ],
                        None,
                        None,
                        None,
                    )

            output_form = output_form.update_properties_to_verify(
                verified,
                falsified,
                query_lbs=bounds,
                true_ub=False,
                ignore_and_falsified=True,
                easiest_and_first=True,
            )

        if verified_so_far >= verification_target:
            return (True, None, None, 0, 0)

        initial_bounds = self.network.get_current_intermediate_bounds()
        # Sort properties by dp bound
        # assert unit_bounds is not None

        unit_clauses = output_form.get_unit_clauses()
        # assert len(unit_clauses) == len(unit_bounds)

        bound_clause = unit_clauses  # list(zip(unit_bounds, unit_clauses))
        # bound_clause.sort(key=lambda x: x[0], reverse=True)

        for clause_id, const_list in bound_clause:
            print(
                f"Verified so far: {verified_so_far} - Target: {verification_target} - Gap: {verification_target -verified_so_far} - Remaining: {remaining}"
            )
            is_verified = False
            for gt_tuple in const_list:

                if time.time() > timeout:
                    print("Verification timed out.")
                    return (
                        verified_so_far >= verification_target,
                        None,
                        const_list,
                        -float("inf"),
                        float("inf"),
                    )

                property_id = f"sample_{str(sample_id)}_gt_tuple_{gt_tuple[0]}_{gt_tuple[1]}_{gt_tuple[2]}"
                time_remaining = timeout - time.time()
                time_budget = max(
                    3,
                    int(
                        min(
                            2
                            * time_remaining
                            / (verification_target - verified_so_far + 1)
                            * remaining
                            / (remaining + verified_so_far - verification_target + 1),
                            time_remaining / min(3, remaining),
                        )
                    ),
                )
                is_verified, best_lb, best_ub, pot_ub_inputs = self.verify_property(
                    property_id,
                    input_lb,
                    input_ub,
                    gt_tuple,
                    time_budget,
                    initial_bounds=initial_bounds,
                )
                remaining -= 1
                if is_verified:
                    verified_so_far += 1
                    break
                else:
                    given_up_on_so_far += 1

            if verified_so_far >= verification_target:
                break

        return (
            verified_so_far >= verification_target,
            None,
            None,
            -float("inf"),
            float("inf"),
        )

    def append_out_adapter(
        self,
        disjunction_adapter: nn.Sequential,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.network.append_out_adapter(disjunction_adapter, device, dtype)
        self.n_output_nodes = self.network.output_dim[0]

    def remove_out_adapter(self) -> None:
        self.network.remove_out_adapter()
        self.n_output_nodes = self.network.output_dim[0]
