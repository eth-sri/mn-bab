import time
from typing import Any, Dict, Sequence, Tuple

import torch
from torch import Tensor

from src.abstract_layers.abstract_network import AbstractNetwork
from src.branch_and_bound import BranchAndBound
from src.exceptions.verification_timeout import VerificationTimeoutException
from src.mn_bab_optimizer import MNBabOptimizer
from src.utilities.attacks import torch_whitebox_attack


class MNBaBVerifier:
    def __init__(
        self,
        network: AbstractNetwork,
        device: torch.device,
        optimize_alpha: bool,
        alpha_lr: float,
        alpha_opt_iterations: int,
        optimize_prima: bool,
        prima_lr: float,
        prima_opt_iterations: int,
        prima_hyperparamters: Dict[str, float],
        peak_lr_scaling_factor: float,
        final_lr_div_factor: float,
        beta_lr: float,
        bab_batch_sizes: Sequence[int],
        branching_config: Dict[str, Any],
        recompute_intermediate_bounds_after_branching: bool,
        use_dependence_sets: bool,
        use_early_termination: bool,
    ) -> None:
        self.network = network
        self.optimizer = MNBabOptimizer(
            optimize_alpha,
            alpha_lr,
            alpha_opt_iterations,
            optimize_prima,
            prima_lr,
            prima_opt_iterations,
            prima_hyperparamters,
            peak_lr_scaling_factor,
            final_lr_div_factor,
            beta_lr,
            use_dependence_sets,
            use_early_termination,
        )
        self.bab = BranchAndBound(
            self.optimizer,
            bab_batch_sizes,
            branching_config,
            recompute_intermediate_bounds_after_branching,
            device,
        )

        assert len(self.network.output_dim) == 1
        self.n_output_nodes = self.network.output_dim[0]

    def _verify_property_with_deep_poly(
        self,
        input_lb: Tensor,
        input_ub: Tensor,
        label: int,
        competing_label: int,
        early_stopping_threshold: float,
    ) -> bool:
        print("Verifying label ", label, " against ", competing_label)
        query_coef = torch.zeros(1, 1, self.n_output_nodes, device=input_lb.device)
        query_coef.data[0, 0, label] = 1
        query_coef.data[0, 0, competing_label] = -1

        deep_poly_lbs, __ = self.bab.optimizer.bound_minimum_with_deep_poly(
            query_coef, self.network, input_lb, input_ub
        )
        print("deep poly lower bounds:", deep_poly_lbs[0])
        return deep_poly_lbs[0] >= early_stopping_threshold

    def verify(
        self,
        sample_id: int,
        input: Tensor,
        input_lb: Tensor,
        input_ub: Tensor,
        label: int,
        timeout: float,
    ) -> bool:
        start_time = time.time()

        def generate_constraints(
            class_num: int, y: int
        ) -> Sequence[Sequence[Tuple[int, int, int]]]:
            return [[(y, i, 0)] for i in range(class_num) if i != y]

        properties_to_verify = []
        for constraint_list in generate_constraints(10, int(label)):
            true_label, competing_label, early_stopping_threshold = constraint_list[0]
            if not self._verify_property_with_deep_poly(
                input_lb,
                input_ub,
                true_label,
                competing_label,
                early_stopping_threshold,
            ):
                properties_to_verify.append(constraint_list)

        if not properties_to_verify:
            return True

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
            return False

        for constraint_list in properties_to_verify:
            true_label, competing_label, early_stopping_threshold = constraint_list[0]

            if time.time() - start_time > timeout:
                print("Verification timed out.")
                return False

            property_id = (
                "sample"
                + str(sample_id)
                + "_label"
                + str(label)
                + "_adversarial_label"
                + str(competing_label)
            )
            time_remaining = timeout - (time.time() - start_time)
            if not self.verify_property(
                property_id,
                input_lb,
                input_ub,
                label,
                competing_label,
                time_remaining,
                early_stopping_threshold,
            ):
                return False

        return True

    def verify_property(
        self,
        property_id: str,
        input_lb: Tensor,
        input_ub: Tensor,
        label: int,
        competing_label: int,
        timeout: float,
        early_stopping_threshold: float,
    ) -> bool:
        print("Verifying label ", label, " against ", competing_label)
        query_coef = torch.zeros(1, 1, self.n_output_nodes)
        query_coef.data[0, 0, label] = 1
        query_coef.data[0, 0, competing_label] = -1

        try:
            property_lb = self.bab.lower_bound_property_with_branch_and_bound(
                property_id,
                query_coef,
                self.network,
                input_lb,
                input_ub,
                early_stopping_threshold=early_stopping_threshold,
                timeout=timeout,
            )
        except VerificationTimeoutException:
            print(
                "Verification of label ",
                label,
                " against ",
                competing_label,
                " timed out.",
            )
            return False

        return property_lb >= early_stopping_threshold
