import os.path
import shutil
import time
from pathlib import Path
from typing import Callable, Tuple

import onnx  # type: ignore [import]
import torch
from torch import Tensor

from src.abstract_layers.abstract_network import AbstractNetwork
from src.utilities.config import make_backsubstitution_config
from src.utilities.initialization import seed_everything
from src.utilities.loading.dnnv_simplify import simplify_onnx
from src.utilities.loading.network import (
    freeze_network,
    load_onnx_from_proto,
    load_onnx_model,
)
from tests.test_util import (  # dpf_call,; get_nn4sys_128d_block,; get_nn4sys_128d_multipath_block_stacked,; get_nn4sys_128d_splitblock,
    abs_toy_pad_net,
    abs_toy_pad_tiny_net,
    dp_call,
    get_deep_poly_bounds,
    get_mnist_net,
    get_relu_lin_layer,
    get_three_relu_lin_layer,
    get_two_relu_lin_layer,
    milp_call,
    prima_crown_wrapper_call,
    run_fuzzing_test,
    toy_max_pool_mixed_net,
)

TEST = run_fuzzing_test

SCRIPT_PATH = os.path.realpath(os.path.dirname(__file__))
# from tests.test_util import toy_max_pool_tiny_net


class TestFuzzing:
    """
    We test with our Fuzzing implementation
    """

    @staticmethod
    def fuzzing_test_network(
        network_constructor: Callable[[], Tuple[AbstractNetwork, Tuple[int, ...]]],
        bounding_call: Callable[
            [AbstractNetwork, Tuple[Tensor, Tensor]], Tuple[Tensor, Tensor]
        ],
        eps: float = 0.1,
        n: int = 20,
        input_domain: Tuple[float, float] = (-1, 1),
    ) -> None:
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        dtype = torch.float64
        torch.set_default_dtype(dtype)
        seed_everything(42)
        network, input_dim = network_constructor()
        network = network.to(device).to(dtype=dtype)
        network.eval()
        freeze_network(network)
        print(f"Testing with eps={eps}.")

        batched_in_shape = (1, *input_dim)
        for i in range(n):
            seed_everything(42 + i)
            network.reset_output_bounds()
            network.reset_input_bounds()
            input = (
                torch.rand(batched_in_shape, device=device) * input_domain[1]
                - (input_domain[0])
                + input_domain[0]
            )
            input_lb = (input - eps)[0]
            input_ub = (input + eps)[0]
            TEST(
                network,
                input,
                input_lb,
                input_ub,
                batched_in_shape,
                bounding_call=bounding_call,
                use_beta=True,
                use_adv=True,
            )

        torch.set_default_dtype(torch.float32)

    def test_maxpool_toy_example(self) -> None:
        self.fuzzing_test_network(toy_max_pool_mixed_net, dp_call, 0.001, 1)
        self.fuzzing_test_network(toy_max_pool_mixed_net, milp_call, 0.001, 1)
        self.fuzzing_test_network(toy_max_pool_mixed_net, dp_call, 0.01, 1)
        self.fuzzing_test_network(toy_max_pool_mixed_net, milp_call, 0.01, 1)

    def test_pad_toy_example(self) -> None:
        self.fuzzing_test_network(abs_toy_pad_net, dp_call, 0.01, 1)
        self.fuzzing_test_network(abs_toy_pad_net, milp_call, 0.01, 1)
        self.fuzzing_test_network(abs_toy_pad_tiny_net, dp_call, 0.01, 1)
        self.fuzzing_test_network(abs_toy_pad_tiny_net, milp_call, 0.01, 1)

    def test_alpha_optimization(self) -> None:
        alpha_call = prima_crown_wrapper_call(optimize_alpha=True, optimize_prima=False)

        self.fuzzing_test_network(
            get_mnist_net,
            dp_call,
            0.001,
            5,
        )

        self.fuzzing_test_network(
            get_mnist_net,
            alpha_call,
            0.001,
            5,
        )

        self.fuzzing_test_network(
            get_mnist_net,
            alpha_call,
            0.01,
            5,
        )

    def test_alpha_prima_optimization(self) -> None:
        alpha_prima_call = prima_crown_wrapper_call(
            optimize_alpha=True, optimize_prima=True
        )

        # self.fuzzing_test_network(
        #     get_mnist_net,
        #     dp_call,
        #     0.01,
        #     1,
        # )

        # self.fuzzing_test_network(
        #     get_mnist_net,
        #     alpha_prima_call,
        #     0.001,
        #     5,
        # )

        # self.fuzzing_test_network(
        #     get_mnist_net,
        #     alpha_prima_call,
        #     0.01,
        #     1,
        # )

        self.fuzzing_test_network(
            get_relu_lin_layer,
            alpha_prima_call,
            0.4,
            10,
        )

        self.fuzzing_test_network(
            get_two_relu_lin_layer,
            alpha_prima_call,
            0.2,
            10,
        )

        self.fuzzing_test_network(
            get_three_relu_lin_layer,
            alpha_prima_call,
            0.2,
            10,
        )
        self.fuzzing_test_network(
            get_three_relu_lin_layer,
            alpha_prima_call,
            0.1,
            5,
        )
        self.fuzzing_test_network(
            get_three_relu_lin_layer,
            alpha_prima_call,
            0.01,
            5,
        )

    def test_nn4sys(self) -> None:

        # alpha_call = prima_crown_wrapper_call(optimize_alpha=True, optimize_prima=False)
        # alpha_prima_call = prima_crown_wrapper_call(
        #     optimize_alpha=True, optimize_prima=True
        # )

        def get_nn4sys_128d() -> Tuple[AbstractNetwork, Tuple[int, ...]]:
            onnx_path = os.path.realpath(
                os.path.join(
                    SCRIPT_PATH,
                    "../../vnn-comp-2022-sup/benchmarks/nn4sys/onnx/mscn_128d.onnx",
                )
            )
            o2p_net, in_shape, in_name = load_onnx_model(onnx_path)
            # o2p_net[1] = o2p_net[1][:-1]
            freeze_network(o2p_net)
            o2p_net.eval()
            abs_net = AbstractNetwork.from_concrete_module(o2p_net, in_shape)
            return abs_net, in_shape

        def get_nn4sys_128d_dual() -> Tuple[AbstractNetwork, Tuple[int, ...]]:
            onnx_path = os.path.realpath(
                os.path.join(
                    SCRIPT_PATH,
                    "../../vnn-comp-2022-sup/benchmarks/nn4sys/onnx/mscn_128d_dual.onnx",
                )
            )
            o2p_net, in_shape, in_name = load_onnx_model(onnx_path)
            # o2p_net[1] = o2p_net[1][:-1]
            freeze_network(o2p_net)
            o2p_net.eval()
            abs_net = AbstractNetwork.from_concrete_module(o2p_net, in_shape)
            return abs_net, in_shape

        def prev_interval_to_call(
            call: Callable[
                [AbstractNetwork, Tuple[Tensor, Tensor]], Tuple[Tensor, Tensor]
            ]
        ) -> Callable[[AbstractNetwork, Tuple[Tensor, Tensor]], Tuple[Tensor, Tensor]]:
            def internal_call(
                net: AbstractNetwork, bounds: Tuple[Tensor, Tensor]
            ) -> Tuple[Tensor, Tensor]:
                bounds = (bounds[0].unsqueeze(0), bounds[1].unsqueeze(0))
                lb, ub = net.set_layer_bounds_via_interval_propagation(
                    bounds[0], bounds[1], use_existing_bounds=False, has_batch_dim=True
                )
                return call(net, bounds)

            return internal_call

        self.fuzzing_test_network(
            get_nn4sys_128d_dual,
            prev_interval_to_call(dp_call),
            0.1,
            5,
            input_domain=(0.1, 2),
        )
        self.fuzzing_test_network(
            get_nn4sys_128d_dual,
            prev_interval_to_call(dp_call),
            0.05,
            5,
            input_domain=(0.1, 2),
        )
        self.fuzzing_test_network(
            get_nn4sys_128d_dual,
            prev_interval_to_call(dp_call),
            0.01,
            5,
            input_domain=(0.1, 2),
        )
        self.fuzzing_test_network(
            get_nn4sys_128d_dual,
            prev_interval_to_call(dp_call),
            0.005,
            5,
            input_domain=(0.1, 2),
        )
        self.fuzzing_test_network(
            get_nn4sys_128d_dual,
            prev_interval_to_call(dp_call),
            0.001,
            5,
            input_domain=(0.1, 2),
        )
        self.fuzzing_test_network(
            get_nn4sys_128d_dual,
            prev_interval_to_call(dp_call),
            0.0005,
            5,
            input_domain=(0.1, 2),
        )
        self.fuzzing_test_network(
            get_nn4sys_128d_dual,
            prev_interval_to_call(dp_call),
            0.0001,
            5,
            input_domain=(0.1, 2),
        )

        # Single
        self.fuzzing_test_network(
            get_nn4sys_128d,
            prev_interval_to_call(dp_call),
            0.1,
            5,
            input_domain=(0.1, 2),
        )
        self.fuzzing_test_network(
            get_nn4sys_128d,
            prev_interval_to_call(dp_call),
            0.05,
            5,
            input_domain=(0.1, 2),
        )
        self.fuzzing_test_network(
            get_nn4sys_128d,
            prev_interval_to_call(dp_call),
            0.01,
            5,
            input_domain=(0.1, 2),
        )
        self.fuzzing_test_network(
            get_nn4sys_128d,
            prev_interval_to_call(dp_call),
            0.005,
            5,
            input_domain=(0.1, 2),
        )
        self.fuzzing_test_network(
            get_nn4sys_128d,
            prev_interval_to_call(dp_call),
            0.001,
            5,
            input_domain=(0.1, 2),
        )
        self.fuzzing_test_network(
            get_nn4sys_128d,
            prev_interval_to_call(dp_call),
            0.0005,
            5,
            input_domain=(0.1, 2),
        )
        self.fuzzing_test_network(
            get_nn4sys_128d,
            prev_interval_to_call(dp_call),
            0.0001,
            5,
            input_domain=(0.1, 2),
        )
        # self.fuzzing_test_network(
        #     get_nn4sys_128d, prev_interval_to_call(alpha_call), 0.01, 1
        # )
        # self.fuzzing_test_network(
        #     get_nn4sys_128d, prev_interval_to_call(alpha_prima_call), 0.03, 1
        # )

    def test_carvana_unet(self) -> None:
        def get_carvana_unet() -> Tuple[AbstractNetwork, Tuple[int, ...]]:
            path = "vnn-comp-2022-sup/benchmarks/carvana_unet_2022/onnx/unet_simp_small.onnx"
            o2p_net, in_shape, _ = load_onnx_model(path)
            o2p_net.eval()
            abs_net = AbstractNetwork.from_concrete_module(o2p_net, in_shape)
            return abs_net, (1, *in_shape)

        def forward_to_dp(
            net: AbstractNetwork, bounds: Tuple[Tensor, Tensor]
        ) -> Tuple[Tensor, Tensor]:
            bounds = (bounds[0], bounds[1])
            bs_config = make_backsubstitution_config(
                use_dependence_sets=False,
                use_early_termination=False,
                max_num_queries=1000,
            )

            with torch.no_grad():
                net.set_layer_bounds_via_forward_dp_pass(
                    bs_config,
                    input_lb=bounds[0],
                    input_ub=bounds[1],
                    timeout=time.time() + 200,
                )
            return get_deep_poly_bounds(
                net,
                bounds[0],
                bounds[1],
                use_dependence_sets=False,
                use_early_termination=False,
                reset_input_bounds=False,
                recompute_intermediate_bounds=False,
                max_num_query=500,
            )

        self.fuzzing_test_network(get_carvana_unet, forward_to_dp, 0.03, 5)

    def test_onnx_simplification(self) -> None:
        def get_tll_onnx() -> Tuple[AbstractNetwork, Tuple[int, ...]]:
            onnx_path = "vnn-comp-2022-sup/benchmarks/tllverifybench/onnx/tllBench_n=2_N=M=24_m=1_instance_2_3.onnx.gz"
            o2p_net, in_shape, in_name = load_onnx_model(onnx_path)
            freeze_network(o2p_net)
            o2p_net.eval()
            o2p_net.to(dtype=torch.get_default_dtype())

            simplify = False
            if simplify:
                assert in_shape is not None
                assert in_name is not None
                # export current model to onnx for dtype
                try:
                    temp_dir = "temp_convert"
                    net_pref = "simplify"
                    onnx_path = f"{temp_dir}/{net_pref}.onnx"
                    Path(temp_dir).mkdir(parents=True, exist_ok=True)
                    x = torch.rand((1, *in_shape), device="cpu")
                    torch.onnx.export(
                        o2p_net,
                        x,
                        onnx_path,
                        export_params=True,
                        training=0,
                        do_constant_folding=True,
                        verbose=False,
                        input_names=[in_name],
                        output_names=["output"],
                    )
                    onnx_model = onnx.load(onnx_path)
                    onnx_model = simplify_onnx(onnx_model)
                    net_new, _, _ = load_onnx_from_proto(onnx_model)
                    o2p_net = net_new
                except Exception as e:
                    print("Exception simplifying onnx model", e)
                finally:
                    shutil.rmtree(temp_dir, ignore_errors=True)

            abs_net = AbstractNetwork.from_concrete_module(o2p_net, in_shape)
            return abs_net, (1, *in_shape)

        self.fuzzing_test_network(get_tll_onnx, dp_call, 10, 1)


if __name__ == "__main__":
    T = TestFuzzing()
    T.test_nn4sys()
    # T.test_alpha_prima_optimization()
    # T.test_maxpool_toy_example()
    # T.test_carvana_unet()
    # T.test_onnx_simplification()
