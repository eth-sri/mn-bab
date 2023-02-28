import csv
from copy import deepcopy

import torch
from gurobipy import GRB  # type: ignore[import]

from src.abstract_layers.abstract_network import AbstractNetwork
from src.utilities.initialization import seed_everything
from src.utilities.loading.data import transform_and_bound
from src.utilities.loading.network import freeze_network, mnist_a_b, mnist_conv_tiny
from tests.gurobi_util import create_milp_model
from tests.test_util import (
    MNIST_CONV_DATA_TEST_CONFIG,
    MNIST_FC_DATA_TEST_CONFIG,
    MNIST_INPUT_DIM,
    get_deep_poly_bounds,
    lower_bound_output_node_with_branch_and_bound,
    optimize_output_node_bounds_with_prima_crown,
)

NUM_EPS = 1e-6


class TestSoundness:
    def test_deep_poly_soundness_on_small_mnist_net(self) -> None:
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
            print("Testing soundness for test instance:", i)
            image, input_lb, input_ub = transform_and_bound(
                pixel_values, MNIST_FC_DATA_TEST_CONFIG
            )

            pred_label = torch.argmax(original_network(image))
            if pred_label != int(label):
                print("Network fails on test image, skipping.")
                continue

            (
                output_lb_deep_poly,
                output_ub_deep_poly,
            ) = get_deep_poly_bounds(network, input_lb, input_ub)
            network.reset_input_bounds()
            network.reset_output_bounds()
            network.set_layer_bounds_via_interval_propagation(input_lb, input_ub)
            model, var_list = create_milp_model(network, input_lb, input_ub)

            output_lb_milp = torch.full(size=(10,), fill_value=0.0, dtype=torch.float64)
            output_ub_milp = torch.full(
                size=(10,), fill_value=float("inf"), dtype=torch.float64
            )
            for j in range(10):
                output_node_var = var_list[-10 + j]
                obj = output_node_var

                model.setObjective(obj, GRB.MINIMIZE)
                model.optimize()
                assert model.status == GRB.OPTIMAL
                output_lb_milp[j] = model.objVal
                model.reset(0)

                model.setObjective(obj, GRB.MAXIMIZE)
                model.optimize()
                assert model.status == GRB.OPTIMAL
                output_ub_milp[j] = model.objVal
                model.reset(0)

            assert (output_lb_deep_poly <= output_lb_milp + NUM_EPS).all()
            assert (output_ub_deep_poly >= output_ub_milp - NUM_EPS).all()
            network.reset_input_bounds()

    def test_prima_crown_alpha_soundness_on_small_mnist_net(self) -> None:
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
            print("Testing soundness for test instance:", i)
            image, input_lb, input_ub = transform_and_bound(
                pixel_values, MNIST_FC_DATA_TEST_CONFIG
            )

            pred_label = torch.argmax(original_network(image))
            if pred_label != int(label):
                print("Network fails on test image, skipping.")
                continue

            output_lb_prima_crown = torch.full(
                size=(10,), fill_value=0.0, dtype=torch.float64
            )
            output_ub_prima_crown = torch.full(
                size=(10,), fill_value=float("inf"), dtype=torch.float64
            )

            output_lb_milp = torch.full(size=(10,), fill_value=0.0, dtype=torch.float64)
            output_ub_milp = torch.full(
                size=(10,), fill_value=float("inf"), dtype=torch.float64
            )
            network.reset_input_bounds()
            network.reset_output_bounds()
            network.set_layer_bounds_via_interval_propagation(input_lb, input_ub)
            model, var_list = create_milp_model(network, input_lb, input_ub)

            for j in range(10):
                output_node_var = var_list[-10 + j]
                obj = output_node_var

                model.setObjective(obj, GRB.MINIMIZE)
                model.optimize()
                assert model.status == GRB.OPTIMAL
                output_lb_milp[j] = model.objVal
                model.reset(0)

                model.setObjective(obj, GRB.MAXIMIZE)
                model.optimize()
                assert model.status == GRB.OPTIMAL
                output_ub_milp[j] = model.objVal
                model.reset(0)

                (
                    prima_crown_alpha_lb,
                    prima_crown_alpha_ub,
                ) = optimize_output_node_bounds_with_prima_crown(
                    network,
                    j,
                    input_lb,
                    input_ub,
                    optimize_alpha=True,
                )
                output_lb_prima_crown[j] = prima_crown_alpha_lb
                output_ub_prima_crown[j] = prima_crown_alpha_ub

            assert (output_lb_prima_crown <= output_lb_milp + NUM_EPS).all()
            assert (output_ub_prima_crown >= output_ub_milp - NUM_EPS).all()

    def test_prima_crown_alpha_prima_soundness_on_small_mnist_net(self) -> None:
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
            print("Testing soundness for test instance:", i)
            image, input_lb, input_ub = transform_and_bound(
                pixel_values, MNIST_FC_DATA_TEST_CONFIG
            )

            pred_label = torch.argmax(original_network(image))
            if pred_label != int(label):
                print("Network fails on test image, skipping.")
                continue

            output_lb_prima_crown = torch.full(
                size=(10,), fill_value=0.0, dtype=torch.float64
            )
            output_ub_prima_crown = torch.full(
                size=(10,), fill_value=float("inf"), dtype=torch.float64
            )

            output_lb_milp = torch.full(size=(10,), fill_value=0.0, dtype=torch.float64)
            output_ub_milp = torch.full(
                size=(10,), fill_value=float("inf"), dtype=torch.float64
            )
            network.reset_input_bounds()
            network.reset_output_bounds()
            network.set_layer_bounds_via_interval_propagation(input_lb, input_ub)
            model, var_list = create_milp_model(network, input_lb, input_ub)

            for j in range(10):
                output_node_var = var_list[-10 + j]
                obj = output_node_var

                model.setObjective(obj, GRB.MINIMIZE)
                model.optimize()
                assert model.status == GRB.OPTIMAL
                output_lb_milp[j] = model.objVal
                model.reset(0)

                model.setObjective(obj, GRB.MAXIMIZE)
                model.optimize()
                assert model.status == GRB.OPTIMAL
                output_ub_milp[j] = model.objVal
                model.reset(0)

                (
                    prima_crown_alpha_prima_lb,
                    prima_crown_alpha_prima_ub,
                ) = optimize_output_node_bounds_with_prima_crown(
                    network,
                    j,
                    input_lb,
                    input_ub,
                    optimize_alpha=True,
                    optimize_prima=True,
                )
                output_lb_prima_crown[j] = prima_crown_alpha_prima_lb
                output_ub_prima_crown[j] = prima_crown_alpha_prima_ub

            assert (output_lb_prima_crown <= output_lb_milp).all()
            assert (output_ub_prima_crown >= output_ub_milp).all()

    def test_prima_crown_alpha_prima_soundness_on_mnist_conv_tiny(self) -> None:
        seed_everything(10)
        num_samples = 10
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

        for i, (label, *pixel_values) in enumerate(test_instances):
            if i >= num_samples:
                break
            print("Testing soundness for test instance:", i)
            image, input_lb, input_ub = transform_and_bound(
                pixel_values, MNIST_CONV_DATA_TEST_CONFIG
            )

            pred_label = torch.argmax(original_network(image))
            if pred_label != int(label):
                print("Network fails on test image, skipping.")
                continue

            output_lb_prima_crown = torch.full(
                size=(10,), fill_value=0.0, dtype=torch.float64
            )
            output_ub_prima_crown = torch.full(
                size=(10,), fill_value=float("inf"), dtype=torch.float64
            )

            output_lb_milp = torch.full(size=(10,), fill_value=0.0, dtype=torch.float64)
            output_ub_milp = torch.full(
                size=(10,), fill_value=float("inf"), dtype=torch.float64
            )
            network.reset_input_bounds()
            network.reset_output_bounds()
            network.set_layer_bounds_via_interval_propagation(input_lb, input_ub)
            model, var_list = create_milp_model(network, input_lb, input_ub)

            for j in range(10):
                output_node_var = var_list[-10 + j]
                obj = output_node_var

                model.setObjective(obj, GRB.MINIMIZE)
                model.optimize()
                assert model.status == GRB.OPTIMAL
                output_lb_milp[j] = model.objVal
                model.reset(0)

                model.setObjective(obj, GRB.MAXIMIZE)
                model.optimize()
                assert model.status == GRB.OPTIMAL
                output_ub_milp[j] = model.objVal
                model.reset(0)

                (
                    prima_crown_alpha_prima_lb,
                    prima_crown_alpha_prima_ub,
                ) = optimize_output_node_bounds_with_prima_crown(
                    network,
                    j,
                    input_lb,
                    input_ub,
                    optimize_alpha=True,
                    optimize_prima=True,
                )
                output_lb_prima_crown[j] = prima_crown_alpha_prima_lb
                output_ub_prima_crown[j] = prima_crown_alpha_prima_ub

            assert (output_lb_prima_crown <= output_lb_milp + NUM_EPS).all()
            assert (output_ub_prima_crown >= output_ub_milp - NUM_EPS).all()

    def test_branch_and_bound_soundness_on_small_mnist_net(self) -> None:
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

        test_config = deepcopy(MNIST_FC_DATA_TEST_CONFIG)
        test_config.eps = 0.005
        tolerance = 1e-5

        for i, (label, *pixel_values) in enumerate(test_instances):
            if i >= num_samples:
                break
            print("Testing soundness for test instance:", i)
            image, input_lb, input_ub = transform_and_bound(pixel_values, test_config)

            pred_label = torch.argmax(original_network(image))
            if pred_label != int(label):
                print("Network fails on test image, skipping.")
                continue

            output_lb_bab = torch.full(size=(10,), fill_value=0.0, dtype=torch.float64)
            output_lb_milp = torch.full(size=(10,), fill_value=0.0, dtype=torch.float64)

            network.reset_input_bounds()
            network.reset_output_bounds()
            network.set_layer_bounds_via_interval_propagation(input_lb, input_ub)
            model, var_list = create_milp_model(network, input_lb, input_ub)

            for j in range(10):
                output_node_var = var_list[-10 + j]
                obj = output_node_var

                model.setObjective(obj, GRB.MINIMIZE)
                model.optimize()
                assert model.status == GRB.OPTIMAL
                output_lb_milp[j] = model.objVal
                model.reset(0)

                # skip the uninteresting cases
                if output_lb_milp[j] != 0:
                    output_lb_bab[j] = lower_bound_output_node_with_branch_and_bound(
                        network,
                        j,
                        input_lb,
                        input_ub,
                        batch_sizes=[4, 4, 4],
                        optimize_alpha=True,
                        optimize_prima=False,
                    )

            assert (output_lb_bab <= output_lb_milp + tolerance).all()

    def test_branch_and_bound_soundness_on_mnist_conv_tiny(self) -> None:
        seed_everything(10)
        num_samples = 10
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

        test_config = deepcopy(MNIST_CONV_DATA_TEST_CONFIG)
        test_config.eps = 0.005
        tolerance = 1e-5

        for i, (label, *pixel_values) in enumerate(test_instances):
            if i >= num_samples:
                break
            print("Testing soundness for test instance:", i)
            image, input_lb, input_ub = transform_and_bound(pixel_values, test_config)

            pred_label = torch.argmax(original_network(image))
            if pred_label != int(label):
                print("Network fails on test image, skipping.")
                continue

            output_lb_bab = torch.full(size=(10,), fill_value=0.0, dtype=torch.float64)
            output_lb_milp = torch.full(size=(10,), fill_value=0.0, dtype=torch.float64)

            network.reset_input_bounds()
            network.reset_output_bounds()
            network.set_layer_bounds_via_interval_propagation(input_lb, input_ub)
            model, var_list = create_milp_model(network, input_lb, input_ub)

            for j in range(10):
                output_node_var = var_list[-10 + j]
                obj = output_node_var

                model.setObjective(obj, GRB.MINIMIZE)
                model.optimize()
                assert model.status == GRB.OPTIMAL
                output_lb_milp[j] = model.objVal
                model.reset(0)

                # skip the uninteresting cases
                if output_lb_milp[j] != 0:
                    output_lb_bab[j] = lower_bound_output_node_with_branch_and_bound(
                        network,
                        j,
                        input_lb,
                        input_ub,
                        batch_sizes=[4, 4, 4],
                        optimize_alpha=True,
                        optimize_prima=False,
                    )

            assert (output_lb_bab <= output_lb_milp + tolerance).all()


if __name__ == "__main__":
    T = TestSoundness()
    # T.test_deep_poly_soundness_on_small_mnist_net()
    T.test_prima_crown_alpha_soundness_on_small_mnist_net()
    T.test_branch_and_bound_soundness_on_small_mnist_net()
