import csv

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
    lower_bound_output_node_with_branch_and_bound,
)

TOL = 1e-3


class TestCompleteness:
    def test_branch_and_bound_completeness_on_small_mnist_net(self) -> None:
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

            output_lb_bab = torch.full(size=(10,), fill_value=0.0, dtype=torch.float64)
            output_lb_milp = torch.full(size=(10,), fill_value=0.0, dtype=torch.float64)

            network.reset_input_bounds()
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
                        early_stopping_threshold=output_lb_milp[j].item() - TOL,
                        optimize_alpha=True,
                        optimize_prima=False,
                    )

            assert (output_lb_bab >= output_lb_milp - TOL).all()

    def test_branch_and_bound_completeness_on_small_mnist_conv_tiny(self) -> None:
        TOL = 5e-3  # BaB-Optimization not optimal with 20 iterations in alpha
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

            output_lb_bab = torch.full(size=(10,), fill_value=0.0, dtype=torch.float64)
            output_lb_milp = torch.full(size=(10,), fill_value=0.0, dtype=torch.float64)

            network.reset_input_bounds()
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
                        early_stopping_threshold=output_lb_milp[j].item() - TOL,
                        optimize_alpha=True,
                        optimize_prima=False,
                    )

            assert (output_lb_bab >= output_lb_milp - TOL).all()


if __name__ == "__main__":
    T = TestCompleteness()
    T.test_branch_and_bound_completeness_on_small_mnist_conv_tiny()
