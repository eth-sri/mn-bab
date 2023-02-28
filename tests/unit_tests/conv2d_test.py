import csv

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from src.abstract_domains.DP_f import DeepPoly_f
from src.abstract_domains.zonotope import HybridZonotope
from src.abstract_layers.abstract_conv2d import Conv2d
from src.abstract_layers.abstract_network import AbstractNetwork
from src.mn_bab_shape import AffineForm, MN_BaB_Shape
from src.state.tags import query_tag
from src.utilities.config import (
    make_backsubstitution_config,
    make_prima_hyperparameters,
)
from src.utilities.loading.data import transform_and_bound
from src.utilities.loading.network import mnist_conv_small
from src.verification_subproblem import SubproblemState
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
        lb = AffineForm(lb_coef)
        ub = AffineForm(2 * lb_coef)
        initial_shape = MN_BaB_Shape(
            query_id=query_tag(layer),
            query_prev_layer=None,
            queries_to_compute=None,
            lb=lb,
            ub=ub,
            unstable_queries=None,
            subproblem_state=None,
        )

        propagated_shape = layer.backsubstitute(
            make_backsubstitution_config(), initial_shape
        )

        assert isinstance(propagated_shape.lb.coef, Tensor)
        assert propagated_shape.lb.coef.shape == (1, np.prod(output_dim)) + input_dim
        assert propagated_shape.lb.bias.shape == (1, np.prod(output_dim))

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
        lb = AffineForm(lb_coef)
        ub = AffineForm(2 * lb_coef)
        initial_shape = MN_BaB_Shape(
            query_id=query_tag(layer),
            query_prev_layer=None,
            queries_to_compute=None,
            lb=lb,
            ub=ub,
            unstable_queries=None,
            subproblem_state=None,
        )

        propagated_shape = layer.backsubstitute(
            make_backsubstitution_config(), initial_shape
        )

        assert isinstance(propagated_shape.lb.coef, Tensor)
        assert propagated_shape.lb.coef.shape == (1, np.prod(output_dim)) + input_dim
        assert propagated_shape.lb.bias.shape == (1, np.prod(output_dim))

    def test_propagate_abs_conv_padding(self) -> None:
        in_channels = 1
        out_channels = 2
        input_dim = (in_channels, 5, 3)

        layer = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            input_dim=input_dim,
            padding=1,
            bias=True,
        )

        x = torch.rand((2, *input_dim))
        x_out = layer(x)

        in_zono = HybridZonotope.construct_from_noise(
            x, eps=0.01, domain="zono", data_range=(-torch.inf, torch.inf)
        )
        out_zono = layer.propagate_abstract_element(in_zono)
        assert out_zono.shape == x_out.shape
        assert out_zono.may_contain_point(x_out)

        in_dpf = DeepPoly_f.construct_from_noise(
            x, eps=0.01, domain="DPF", data_range=(-torch.inf, torch.inf)
        )
        out_dpf = layer.propagate_abstract_element(in_dpf)
        assert out_dpf.shape == x_out.shape
        assert out_dpf.may_contain_point(x_out)

    def test_propagate_abs_conv_no_padding(self) -> None:
        in_channels = 1
        out_channels = 2
        input_dim = (in_channels, 5, 3)

        layer = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            input_dim=input_dim,
            padding=0,
            bias=True,
        )

        x = torch.rand((2, *input_dim)) * 2 - 1
        x_out = layer(x)

        in_zono = HybridZonotope.construct_from_noise(
            x, eps=0.01, domain="zono", data_range=(-torch.inf, torch.inf)
        )
        out_zono = layer.propagate_abstract_element(in_zono)
        assert out_zono.shape == x_out.shape
        assert out_zono.may_contain_point(x_out), "Bound violation found for Zono!"

        in_dpf = DeepPoly_f.construct_from_noise(
            x, eps=0.01, domain="DPF", data_range=(-torch.inf, torch.inf)
        )
        out_dpf = layer.propagate_abstract_element(in_dpf)
        assert out_dpf.shape == x_out.shape
        assert out_dpf.may_contain_point(x_out), "Bound violation found for DPF!"

    def test_propagate_abs_conv_pos(self) -> None:
        in_channels = 1
        out_channels = 2
        input_dim = (in_channels, 5, 3)

        layer = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            input_dim=input_dim,
            padding=0,
            bias=False,
        )

        layer.weight.data = layer.weight.data.abs()

        x = torch.rand((2, *input_dim)) * 2
        x_out = layer(x)

        in_zono = HybridZonotope.construct_from_noise(
            x, eps=0.01, domain="zono", data_range=(-torch.inf, torch.inf)
        )
        out_zono = layer.propagate_abstract_element(in_zono)
        assert out_zono.shape == x_out.shape
        assert out_zono.may_contain_point(x_out)
        assert (out_zono.concretize()[0] >= 0).all()

        in_dpf = DeepPoly_f.construct_from_noise(
            x, eps=0.01, domain="DPF", data_range=(-torch.inf, torch.inf)
        )
        out_dpf = layer.propagate_abstract_element(in_dpf)
        assert out_dpf.shape == x_out.shape
        assert out_dpf.may_contain_point(x_out)
        assert (out_dpf.concretize()[0] >= 0).all()

    def test_propagate_interval_identity_layer(self) -> None:
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

    def test_propagate_interval_identity_layer_with_bias(self) -> None:
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
        assert layer.bias is not None
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
        nn.init.uniform_(layer.weight, a=0.0, b=1.0)
        assert layer.bias is not None
        nn.init.uniform_(layer.bias, a=0.0, b=1.0)
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
        network.get_mn_bab_shape(
            config=make_backsubstitution_config(),
            input_lb=input_lb,
            input_ub=input_ub,
            query_id=query_tag(network),
            query_coef=initial_bound_coef,
            subproblem_state=SubproblemState.create_default(
                split_state=None,
                optimize_prima=False,
                batch_size=1,
                device=initial_bound_coef.device,
                use_params=True,
            ),
            compute_upper_bound=True,
            reset_input_bounds=True,
            recompute_intermediate_bounds=True,
            optimize_intermediate_bounds=False,
        )

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
        prima_hyperparameters = make_prima_hyperparameters()
        layer_ids_for_which_to_compute_prima_constraints = (
            network.get_activation_layer_ids()
        )
        network.get_mn_bab_shape(
            make_backsubstitution_config().with_prima(
                prima_hyperparameters,
                layer_ids_for_which_to_compute_prima_constraints,
            ),
            input_lb,
            input_ub,
            query_id=query_tag(network),
            query_coef=initial_bound_coef,
            subproblem_state=SubproblemState.create_default(
                split_state=None,
                optimize_prima=True,
                batch_size=1,
                device=initial_bound_coef.device,
                use_params=True,
            ),
            compute_upper_bound=True,
            reset_input_bounds=True,
            recompute_intermediate_bounds=True,
            optimize_intermediate_bounds=False,
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


if __name__ == "__main__":
    T = TestConv2d()
    T.test_propagate_abs_conv_pos()
    T.test_propagate_abs_conv_no_padding()
    T.test_propagate_abs_conv_padding()
    T.test_propagate_interval_identity_layer_with_bias()
    T.test_small_cnn_forward_pass_does_not_crash()
    T.test_propagate_interval_identity_layer()
    T.test_backsubstitution_mn_bab_shape_with_padding()
    T.test_propagate_positive_interval_through_positive_conv_layer()
    T.test_small_cnn_backsubstitution_pass_with_prima_constraints_does_not_crash()
