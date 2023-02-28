import csv

import torch

from src.abstract_layers.abstract_network import AbstractNetwork
from src.utilities.initialization import seed_everything
from src.utilities.loading.data import transform_and_bound
from src.utilities.loading.network import freeze_network, load_onnx_model, mnist_a_b
from tests.test_util import (
    MNIST_FC_DATA_TEST_CONFIG,
    get_deep_poly_bounds,
    opt_intermediate_bounds,
    optimize_output_node_bounds_with_prima_crown,
)


class TestAlphaIntermediateOptimization:
    """These tests currently do not assert anything besides the fact that the code isrunning without a crash."""

    def test_small_mnist_net(self) -> None:

        seed_everything(10)
        num_samples = 5

        network_path = "networks/mnist_2_50_flattened.pyt"

        test_data_path = "test_data/mnist_test_100.csv"
        test_file = open(test_data_path, "r")
        test_instances = csv.reader(test_file, delimiter=",")

        for i, (label, *pixel_values) in enumerate(test_instances):
            if i >= num_samples:
                break
            original_network = mnist_a_b(5, 100)
            state_dict = torch.load(network_path)
            original_network.load_state_dict(state_dict)
            pre_net = original_network[:-1]
            network = AbstractNetwork.from_concrete_module(original_network, (784,))
            pre_network = AbstractNetwork.from_concrete_module(pre_net, (784,))
            freeze_network(network)
            freeze_network(pre_network)

            print("Testing test instance:", i)
            MNIST_FC_DATA_TEST_CONFIG.eps = 0.03
            image, input_lb, input_ub = transform_and_bound(
                pixel_values, MNIST_FC_DATA_TEST_CONFIG
            )

            pred_label = torch.argmax(original_network(image))
            if pred_label != int(label):
                print("Network fails on test image, skipping.")
                continue

            # milp_model = MILPNetwork.build_model_from_abstract_net(
            #    image, input_lb, input_ub, network
            # )
            # lbs, ubs = milp_model.get_network_bounds_at_layer_multi(layer_tag(network.layers[4]), 100, 300, time.time())
            dp_lb, dp_ub = get_deep_poly_bounds(pre_network, input_lb, input_ub)

            opt_lb, opt_ub = optimize_output_node_bounds_with_prima_crown(
                pre_network,
                int(label),
                input_lb,
                input_ub,
                optimize_alpha=True,
                optimize_prima=True,
            )

            opt_intermediate_bounds(
                pre_network,
                input_lb.view((1, -1)),
                input_ub.view((1, -1)),
                use_prima=True,
            )

            double_opt_lb, double_opt_ub = optimize_output_node_bounds_with_prima_crown(
                pre_network,
                int(label),
                input_lb,
                input_ub,
                optimize_alpha=True,
                optimize_prima=True,
            )

            print(f"======= {dp_lb[0][pred_label]} - {opt_lb} - {double_opt_lb}")
            # Get last layer improvement

    def test_resnet_onnx(self) -> None:
        seed_everything(42)
        o2p_net = load_onnx_model(
            "benchmarks_vnn21/cifar10_resnet/onnx/resnet_2b.onnx"
        )[0]
        freeze_network(o2p_net)
        abs_net = AbstractNetwork.from_concrete_module(o2p_net, (3, 32, 32))
        x = torch.rand((1, 3, 32, 32))
        eps = 20 / 255
        input_lb = x - eps
        input_ub = x + eps

        out_shape = opt_intermediate_bounds(abs_net, input_lb, input_ub, use_prima=True)
        final_lb, final_ub = out_shape.concretize(input_lb, input_ub)
        print(f"Mean: {torch.mean(final_ub-final_lb)}")


if __name__ == "__main__":
    t = TestAlphaIntermediateOptimization()
    # t.test_small_mnist_net()
    t.test_resnet_onnx()
