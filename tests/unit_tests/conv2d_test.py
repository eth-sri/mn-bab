import csv

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from src.abstract_layers.abstract_conv2d import Conv2d
from src.abstract_layers.abstract_network import AbstractNetwork
from src.mn_bab_optimizer import DEFAULT_PRIMA_HYPERPARAMETERS
from src.mn_bab_shape import MN_BaB_Shape
from src.utilities.loading.data import transform_and_bound
from src.utilities.loading.network import mnist_conv_small
from tests.test_util import MNIST_CONV_DATA_TEST_CONFIG, MNIST_INPUT_DIM


class TestConv2d:
    def test_backsubstitution_mn_bab_shape_with_padding(self) -> None:
        in_channels = 1
        out_channels = 1
        input_dim = (in_channels, 5, 3)
        output_dim = (out_channels, 5, 3)

        layer = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            input_dim=input_dim,
            padding=1,
            bias=True,
        )

        lb_coef = torch.eye(np.prod(output_dim)).view(-1, *output_dim).unsqueeze(0)
        ub_coef = 2 * lb_coef
        initial_shape = MN_BaB_Shape(lb_coef, ub_coef)

        propagated_shape = layer.backsubstitute(initial_shape)

        assert isinstance(propagated_shape.lb_coef, Tensor)
        assert propagated_shape.lb_coef.shape == (1, np.prod(output_dim)) + input_dim
        assert propagated_shape.lb_bias.shape == (1, np.prod(output_dim))

    def test_backsubstitution_mn_bab_shape_without_padding(self) -> None:
        in_channels = 1
        out_channels = 1
        input_dim = (in_channels, 5, 3)
        output_dim = (out_channels, 3, 1)

        layer = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            input_dim=input_dim,
            padding=0,
            bias=True,
        )

        lb_coef = torch.eye(np.prod(output_dim)).view(-1, *output_dim).unsqueeze(0)
        ub_coef = 2 * lb_coef
        initial_shape = MN_BaB_Shape(lb_coef, ub_coef)

        propagated_shape = layer.backsubstitute(initial_shape)

        assert isinstance(propagated_shape.lb_coef, Tensor)
        assert propagated_shape.lb_coef.shape == (1, np.prod(output_dim)) + input_dim
        assert propagated_shape.lb_bias.shape == (1, np.prod(output_dim))

    def test_propagte_interval_identity_layer(self) -> None:
        in_channels = 1
        input_dim = (in_channels, 3, 3)
        layer = Conv2d(
            in_channels=in_channels,
            out_channels=1,
            kernel_size=3,
            input_dim=input_dim,
            padding=1,
            bias=False,
        )
        nn.init.dirac_(layer.weight.data)

        input_lb = torch.full(size=input_dim, fill_value=-1.0).unsqueeze(0)
        input_ub = torch.full(size=input_dim, fill_value=1.0).unsqueeze(0)

        output_lb, output_ub = layer.propagate_interval((input_lb, input_ub))

        assert (output_lb == input_lb).all()
        assert (output_ub == input_ub).all()

    def test_propagte_interval_identity_layer_with_bias(self) -> None:
        in_channels = 1
        input_dim = (in_channels, 3, 3)
        layer = Conv2d(
            in_channels=in_channels,
            out_channels=1,
            kernel_size=3,
            input_dim=input_dim,
            padding=1,
            bias=True,
        )
        nn.init.dirac_(layer.weight.data)
        nn.init.constant_(layer.bias, 0)

        input_lb = torch.full(size=input_dim, fill_value=-1.0).unsqueeze(0)
        input_ub = torch.full(size=input_dim, fill_value=1.0).unsqueeze(0)

        output_lb, output_ub = layer.propagate_interval((input_lb, input_ub))

        assert (output_lb == input_lb).all()
        assert (output_ub == input_ub).all()

    def test_propagate_positive_interval_through_positive_conv_layer(self) -> None:
        test_lb = torch.abs(torch.rand(MNIST_INPUT_DIM).unsqueeze(0))
        test_ub = test_lb + 0.1
        assert (test_lb >= 0).all()

        layer = Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            input_dim=MNIST_INPUT_DIM,
            padding=1,
            bias=True,
        )
        nn.init.uniform(layer.weight, a=0.0, b=1.0)
        nn.init.uniform(layer.bias, a=0.0, b=1.0)
        assert (layer.weight >= 0).all()
        assert (layer.bias >= 0).all()

        expected_output_lb = layer.forward(test_lb)
        expected_output_ub = layer.forward(test_ub)

        after = layer.propagate_interval((test_lb, test_ub))
        assert (after[0] <= after[1]).all()

        assert (expected_output_lb == after[0]).all()
        assert (expected_output_ub == after[1]).all()

    def test_small_cnn_backsubstitution_pass_does_not_crash(self) -> None:
        network_path = "networks/mnist_convSmallRELU__Point.pyt"

        original_network = mnist_conv_small()
        state_dict = torch.load(network_path)
        original_network.load_state_dict(state_dict)
        network = AbstractNetwork.from_concrete_module(
            original_network, MNIST_INPUT_DIM
        )

        test_data_path = "test_data/mnist_test_100.csv"
        test_file = open(test_data_path, "r")
        test_instances = csv.reader(test_file, delimiter=",")

        label, *pixel_values = next(test_instances)
        __, input_lb, input_ub = transform_and_bound(
            pixel_values, MNIST_CONV_DATA_TEST_CONFIG
        )

        initial_bound_coef = (
            torch.eye(np.prod(network.output_dim))
            .view(-1, *network.output_dim)
            .unsqueeze(0)
        )
        abstract_shape = MN_BaB_Shape(
            lb_coef=initial_bound_coef,
            ub_coef=initial_bound_coef,
            carried_over_optimizable_parameters={},
        )
        network.get_mn_bab_shape(input_lb, input_ub, abstract_shape)

    def test_small_cnn_backsubstitution_pass_with_prima_constraints_does_not_crash(
        self,
    ) -> None:
        network_path = "networks/mnist_convSmallRELU__Point.pyt"

        original_network = mnist_conv_small()
        state_dict = torch.load(network_path)
        original_network.load_state_dict(state_dict)
        network = AbstractNetwork.from_concrete_module(
            original_network, MNIST_INPUT_DIM
        )

        test_data_path = "test_data/mnist_test_100.csv"
        test_file = open(test_data_path, "r")
        test_instances = csv.reader(test_file, delimiter=",")

        label, *pixel_values = next(test_instances)
        __, input_lb, input_ub = transform_and_bound(
            pixel_values, MNIST_CONV_DATA_TEST_CONFIG
        )

        initial_bound_coef = (
            torch.eye(np.prod(network.output_dim))
            .view(-1, *network.output_dim)
            .unsqueeze(0)
        )
        abstract_shape = MN_BaB_Shape(
            lb_coef=initial_bound_coef,
            ub_coef=initial_bound_coef,
            carried_over_optimizable_parameters={},
            prima_hyperparamters=DEFAULT_PRIMA_HYPERPARAMETERS,
        )
        layer_ids_for_which_to_compute_prima_constraints = (
            network.get_activation_layer_ids()
        )
        network.get_mn_bab_shape(
            input_lb,
            input_ub,
            abstract_shape,
            layer_ids_for_which_to_compute_prima_constraints=layer_ids_for_which_to_compute_prima_constraints,
        )

    def test_small_cnn_forward_pass_does_not_crash(self) -> None:
        network_path = "networks/mnist_convSmallRELU__Point.pyt"

        original_network = mnist_conv_small()
        state_dict = torch.load(network_path)
        original_network.load_state_dict(state_dict)
        network = AbstractNetwork.from_concrete_module(
            original_network, MNIST_INPUT_DIM
        )

        test_data_path = "test_data/mnist_test_100.csv"
        test_file = open(test_data_path, "r")
        test_instances = csv.reader(test_file, delimiter=",")

        label, *pixel_values = next(test_instances)
        __, input_lb, input_ub = transform_and_bound(
            pixel_values, MNIST_CONV_DATA_TEST_CONFIG
        )

        network.set_layer_bounds_via_interval_propagation(input_lb, input_ub)
