import csv
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from src.abstract_layers.abstract_network import AbstractNetwork
from src.mn_bab_optimizer import MNBabOptimizer
from src.utilities.config import make_backsubstitution_config, make_optimizer_config
from src.utilities.initialization import seed_everything
from src.utilities.loading.data import transform_and_bound
from src.utilities.loading.network import (
    cifar10_cnn_A,
    freeze_network,
    mnist_conv_small,
    resnet2b,
)
from src.verification_subproblem import SubproblemState
from tests.test_util import (
    CIFAR10_CONV_DATA_TEST_CONFIG,
    CIFAR10_INPUT_DIM,
    MNIST_CONV_DATA_TEST_CONFIG,
    MNIST_INPUT_DIM,
)


class TestDependenceSets:
    def _get_bounds_with_all_methods(
        self,
        network: AbstractNetwork,
        optimizer: MNBabOptimizer,
        query_coef: Tensor,
        input_lb: Tensor,
        input_ub: Tensor,
        use_dependence_sets: bool,
    ) -> Dict[str, Tuple[Sequence[float], Sequence[float]]]:
        network.reset_input_bounds()
        batch_size = query_coef.shape[0]
        subproblem_state = SubproblemState.create_default(
            split_state=None,
            optimize_prima=True,
            batch_size=batch_size,
            device=torch.device("cpu"),
            use_params=True,
        )
        dp_lbs, dp_ubs, _, _ = optimizer.bound_minimum_with_deep_poly(
            optimizer.backsubstitution_config,
            input_lb,
            input_ub,
            network,
            query_coef,
            subproblem_state=subproblem_state,
            reset_input_bounds=True,
        )
        assert isinstance(dp_lbs, Sequence)
        assert isinstance(dp_ubs, Sequence)
        alpha_lbs, alpha_ubs, _ = optimizer._bound_minimum_optimizing_alpha(
            optimizer.backsubstitution_config,
            subproblem_state,
            input_lb,
            input_ub,
            network,
            query_coef,
            opt_iterations=optimizer.config.alpha.opt_iterations,
        )
        prima_lbs, prima_ubs, _ = optimizer._bound_minimum_optimizing_alpha_prima(
            optimizer.backsubstitution_config,
            subproblem_state,
            input_lb,
            input_ub,
            network,
            query_coef,
            opt_iterations=optimizer.config.prima.opt_iterations,
        )
        return {
            "deeppoly": (dp_lbs, dp_ubs),
            "alpha": (alpha_lbs, alpha_ubs),
            "prima": (prima_lbs, prima_ubs),
        }

    def test_conv_small(self) -> None:
        seed_everything(1)
        test_pad = False
        test_asym_pad = False
        test_res = False

        assert int(test_res) + int(test_pad) + int(test_asym_pad) <= 1

        using_cifar = test_pad or test_res

        if test_pad:
            network_path = (
                "networks/cifar10_CNN_A_CIFAR_MIX.pyt"  # Has non-zero padding
            )
            original_network = cifar10_cnn_A()
        elif test_res:
            network_path = "networks/resnet_2b.pth"  # Has non-zero padding
            original_network = resnet2b()
        else:
            network_path = "networks/mnist_convSmallRELU__Point.pyt"
            original_network = mnist_conv_small()

        state_dict = torch.load(network_path)
        original_network.load_state_dict(state_dict)
        network = AbstractNetwork.from_concrete_module(
            original_network, CIFAR10_INPUT_DIM if using_cifar else MNIST_INPUT_DIM
        )
        freeze_network(network)

        if using_cifar:
            test_data_path = "test_data/cifar10_test_100.csv"
        else:
            test_data_path = "test_data/mnist_test_100.csv"
        test_file = open(test_data_path, "r")
        test_instances = csv.reader(test_file, delimiter=",")

        optimizer = MNBabOptimizer(
            make_optimizer_config(
                optimize_alpha=True,
                optimize_prima=True,
            ),
            make_backsubstitution_config(),
        )

        batch_size = 3
        for i, (label, *pixel_values) in enumerate(test_instances):
            print(f"Testing test instance: {i}/5")
            if i == 5:
                break
            _, input_lb, input_ub = transform_and_bound(
                pixel_values,
                CIFAR10_CONV_DATA_TEST_CONFIG
                if using_cifar
                else MNIST_CONV_DATA_TEST_CONFIG,
            )
            # batch_size-1 random queries
            query_coef = torch.randint(
                low=-10,
                high=11,
                size=(batch_size, 1, network.output_dim[0]),
                dtype=torch.float,
            )
            # 1 verification query
            query_coef.data[0, 0] = 0
            query_coef.data[0, 0, int(label)] = 1
            query_coef.data[0, 0, 9 - int(label)] = -1
            bounds_old = self._get_bounds_with_all_methods(
                network,
                optimizer,
                query_coef,
                input_lb,
                input_ub,
                use_dependence_sets=False,
            )
            bounds_new = self._get_bounds_with_all_methods(
                network,
                optimizer,
                query_coef,
                input_lb,
                input_ub,
                use_dependence_sets=True,
            )
            for method in ["deeppoly", "alpha", "prima"]:
                for i in range(2):
                    curr_bounds_old = np.asarray(bounds_old[method][i])
                    curr_bounds_new = np.asarray(bounds_new[method][i])
                    print(f"[{method}] {curr_bounds_old} vs {curr_bounds_new}")
                    assert np.isclose(
                        curr_bounds_old, curr_bounds_new, rtol=1e-5, atol=1e-6
                    ).all()
                    # rel_diff = (curr_bounds_old - curr_bounds_new) / curr_bounds_old
                    # assert np.max(np.abs(rel_diff)) < 1e-5, f"{np.max(np.abs(rel_diff))}"
