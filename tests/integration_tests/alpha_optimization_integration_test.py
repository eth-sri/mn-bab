import csv

import torch

from src.abstract_layers.abstract_network import AbstractNetwork
from src.utilities.initialization import seed_everything
from src.utilities.loading.data import transform_and_bound
from src.utilities.loading.network import freeze_network, mnist_a_b, mnist_conv_small
from tests.test_util import (
    MNIST_CONV_DATA_TEST_CONFIG,
    MNIST_FC_DATA_TEST_CONFIG,
    MNIST_INPUT_DIM,
    get_deep_poly_bounds,
    optimize_output_node_bounds_with_prima_crown,
)


class TestAlphaOptimization:
    def test_small_mnist_net(self) -> None:

        seed_everything(10)
        num_samples = 10

        network_path = "networks/mnist_2_50_flattened.pyt"

        original_network = mnist_a_b(2, 50)
        state_dict = torch.load(network_path)
        original_network.load_state_dict(state_dict)
        network = AbstractNetwork.from_concrete_module(original_network, (784,))
        freeze_network(network)

        test_data_path = "test_data/mnist_test_100.csv"
        test_file = open(test_data_path, "r")
        test_instances = csv.reader(test_file, delimiter=",")

        for i, (label, *pixel_values) in enumerate(test_instances):
            if i >= num_samples:
                break
            print("Testing test instance:", i)
            image, input_lb, input_ub = transform_and_bound(
                pixel_values, MNIST_FC_DATA_TEST_CONFIG
            )

            pred_label = torch.argmax(original_network(image))
            if pred_label != int(label):
                print("Network fails on test image, skipping.")
                continue

            (
                output_lb_without_alpha,
                output_ub_without_alpha,
            ) = get_deep_poly_bounds(network, input_lb, input_ub)

            output_lb_with_alpha = torch.full(
                size=(10,), fill_value=0.0, dtype=torch.float64
            )
            output_ub_with_alpha = torch.full(
                size=(10,), fill_value=float("inf"), dtype=torch.float64
            )

            for j in range(10):
                (
                    prima_crown_alpha_lb,
                    prima_crown_alpha_ub,
                ) = optimize_output_node_bounds_with_prima_crown(
                    network, j, input_lb, input_ub, optimize_alpha=True
                )
                output_lb_with_alpha[j] = prima_crown_alpha_lb
                output_ub_with_alpha[j] = prima_crown_alpha_ub

            rounding_error_margin = 1e-5
            assert (
                output_lb_with_alpha + rounding_error_margin >= output_lb_without_alpha
            ).all()
            assert (
                output_ub_with_alpha - rounding_error_margin <= output_ub_without_alpha
            ).all()

    def test_small_cnn(self) -> None:
        seed_everything(10)
        num_samples = 10
        network_path = "networks/mnist_convSmallRELU__Point.pyt"

        original_network = mnist_conv_small()
        state_dict = torch.load(network_path)
        original_network.load_state_dict(state_dict)
        network = AbstractNetwork.from_concrete_module(
            original_network, MNIST_INPUT_DIM
        )
        freeze_network(network)

        test_data_path = "test_data/mnist_test_100.csv"
        test_file = open(test_data_path, "r")
        test_instances = csv.reader(test_file, delimiter=",")

        for i, (label, *pixel_values) in enumerate(test_instances):
            if i >= num_samples:
                break
            print("Testing test instance:", i)
            image, input_lb, input_ub = transform_and_bound(
                pixel_values, MNIST_CONV_DATA_TEST_CONFIG
            )

            (
                output_lb_without_alpha,
                output_ub_without_alpha,
            ) = get_deep_poly_bounds(network, input_lb, input_ub)

            output_lb_with_alpha = torch.full(
                size=(10,), fill_value=0.0, dtype=torch.float64
            )
            output_ub_with_alpha = torch.full(
                size=(10,), fill_value=float("inf"), dtype=torch.float64
            )

            for j in range(10):
                (
                    prima_crown_alpha_lb,
                    prima_crown_alpha_ub,
                ) = optimize_output_node_bounds_with_prima_crown(
                    network, j, input_lb, input_ub, optimize_alpha=True
                )
                output_lb_with_alpha[j] = prima_crown_alpha_lb
                output_ub_with_alpha[j] = prima_crown_alpha_ub

            rounding_error_margin = 1e-4
            assert (
                output_lb_with_alpha + rounding_error_margin >= output_lb_without_alpha
            ).all()
            assert (
                output_ub_with_alpha - rounding_error_margin <= output_ub_without_alpha
            ).all()


if __name__ == "__main__":
    T = TestAlphaOptimization()
    T.test_small_mnist_net()
