import csv
from copy import deepcopy

import torch

from src.abstract_layers.abstract_network import AbstractNetwork
from src.branch_and_bound import BranchAndBound
from src.mn_bab_optimizer import MNBabOptimizer
from src.utilities.config import make_verifier_config
from src.utilities.loading.data import transform_and_bound
from src.utilities.loading.network import freeze_network, mnist_a_b, mnist_conv_tiny
from tests.test_util import (
    MNIST_CONV_DATA_TEST_CONFIG,
    MNIST_FC_DATA_TEST_CONFIG,
    MNIST_INPUT_DIM,
)


class TestBatchProcessing:
    def test_batch_with_bab(self) -> None:
        network_path = "networks/mnist_2_50_flattened.pyt"

        original_network = mnist_a_b(2, 50)
        state_dict = torch.load(network_path)
        original_network.load_state_dict(state_dict)
        network = AbstractNetwork.from_concrete_module(original_network, (784,))
        freeze_network(network)

        test_data_path = "test_data/mnist_test_100.csv"
        test_file = open(test_data_path, "r")
        test_instances = csv.reader(test_file, delimiter=",")

        label_as_string, *pixel_values = next(test_instances)
        label = int(label_as_string)

        test_config = deepcopy(MNIST_FC_DATA_TEST_CONFIG)
        test_config.eps = 0.003
        image, input_lb, input_ub = transform_and_bound(pixel_values, test_config)

        pred_label = torch.argmax(original_network(image))
        assert pred_label == label
        competing_label = 1
        assert label != competing_label

        query_coef = torch.zeros(1, 1, *network.output_dim)
        query_coef.data[:, 0, label] = 1
        query_coef.data[0, 0, competing_label] = -1

        config = make_verifier_config(
            optimize_alpha=True,
            optimize_prima=True,
            bab_batch_sizes=[4, 4],
            recompute_intermediate_bounds_after_branching=True,
        )

        optimizer = MNBabOptimizer(config.optimizer, config.backsubstitution)
        bab = BranchAndBound(
            optimizer, config.bab, config.backsubstitution, torch.device("cpu")
        )

        bab.bound_minimum_with_branch_and_bound(
            "dummy_id", query_coef, network, input_lb, input_ub
        )

    def test_batch_with_bab_with_conv(self) -> None:
        network_path = "networks/mnist_convTiny.pyt"

        original_network = mnist_conv_tiny()
        state_dict = torch.load(network_path)
        original_network.load_state_dict(state_dict)
        network = AbstractNetwork.from_concrete_module(
            original_network, MNIST_INPUT_DIM
        )
        freeze_network(network)

        test_data_path = "test_data/mnist_test_100.csv"
        test_file = open(test_data_path, "r")
        test_instances = csv.reader(test_file, delimiter=",")

        label_as_string, *pixel_values = next(test_instances)
        label = int(label_as_string)

        test_config = deepcopy(MNIST_CONV_DATA_TEST_CONFIG)
        test_config.eps = 0.002
        image, input_lb, input_ub = transform_and_bound(pixel_values, test_config)

        pred_label = torch.argmax(original_network(image))
        assert pred_label == label
        competing_label = 1
        assert label != competing_label

        query_coef = torch.zeros(1, 1, *network.output_dim)
        query_coef.data[:, 0, label] = 1
        query_coef.data[0, 0, competing_label] = -1

        config = make_verifier_config(
            optimize_alpha=True,
            optimize_prima=True,
            bab_batch_sizes=[4, 4, 4],
            recompute_intermediate_bounds_after_branching=True,
        )

        optimizer = MNBabOptimizer(config.optimizer, config.backsubstitution)
        bab = BranchAndBound(
            optimizer, config.bab, config.backsubstitution, torch.device("cpu")
        )

        bab.bound_minimum_with_branch_and_bound(
            "dummy_id", query_coef, network, input_lb, input_ub
        )


if __name__ == "__main__":
    T = TestBatchProcessing()
    T.test_batch_with_bab()
    T.test_batch_with_bab_with_conv()
