from typing import Callable, List, Optional, Tuple, Union

import torch

from src.abstract_layers.abstract_network import AbstractNetwork

# from src.utilities.config import AbstractDomain
from src.utilities.initialization import seed_everything
from src.utilities.loading.network import freeze_network
from tests.test_util import (  # toy_all_layer_net_1d,
    abs_toy_pad_net,
    abs_toy_pad_tiny_net,
    dp_call,
    dpf_call,
    get_convtranspose2d_conv_net,
    milp_call,
    run_fuzzing_test,
    splitting_call,
    toy_all_layer_net,
    toy_avg_pool_net,
    toy_convtranspose2d_net,
    toy_max_pool_mixed_net,
    toy_net,
    toy_sig_net,
    toy_sig_tanh_net,
    zono_call,
)

# from tests.test_util import toy_max_pool_tiny_net


class TestFuzzing:
    """
    We test with our Fuzzing implementation
    """

    @staticmethod
    def fuzzing_test_network(
        network_constructor: Callable[[], Tuple[AbstractNetwork, Tuple[int, ...]]],
        eps: Union[float, List[float]] = 0.1,
        n: int = 20,
        dp: bool = True,
        milp: bool = True,
        dpf: bool = False,
        zono: bool = False,
        input_splitting: bool = False,
        splitting_dict: Optional[dict] = None,
    ) -> None:
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        print(f"Running Test with{network_constructor} using {device}")
        old_dtype = torch.get_default_dtype()
        dtype = torch.float64
        torch.set_default_dtype(dtype)
        seed_everything(42)
        network, input_dim = network_constructor()
        network = network.to(device).to(dtype)
        network.eval()
        freeze_network(network)
        print(f"Testing with eps={eps}.")
        if not isinstance(eps, list):
            eps = [eps]

        in_shape = (1, *input_dim)
        for e in eps:
            for i in range(n):
                seed_everything(42 + i)
                input = torch.rand(in_shape, device=device, dtype=dtype) * 2 - 1
                tight_mask = torch.rand(input.shape, device=device) > 0.6
                if (~tight_mask).sum() == 0:
                    tight_mask = torch.zeros_like(tight_mask)
                input_lb = input - e
                input_ub = torch.where(tight_mask, input - e, input + e)

                bounding_calls = []
                if dp:
                    bounding_calls += [dp_call]
                if dpf:
                    bounding_calls += [dpf_call]
                if zono:
                    bounding_calls += [zono_call]
                if milp:
                    bounding_calls += [milp_call]
                if input_splitting and splitting_dict is not None:
                    bounding_calls += [
                        lambda net, bounds: splitting_call(net, bounds, splitting_dict)  # type: ignore # does not realize splitting_dict is not None
                    ]

                for bounding_call in bounding_calls:
                    network.reset_output_bounds()
                    network.reset_input_bounds()
                    run_fuzzing_test(
                        network,
                        input,
                        input_lb,
                        input_ub,
                        in_shape,
                        bounding_call,
                        use_beta=True,
                        use_adv=True,
                    )

        torch.set_default_dtype(old_dtype)

    def test_maxpool_toy_example(self, n: int = 1) -> None:
        self.fuzzing_test_network(
            toy_max_pool_mixed_net, [0.0001, 0.001, 0.01, 0.05], n, dp=True, dpf=True
        )
        # self.fuzzing_test_network(toy_max_pool_mixed_net, [0.1, 0.5], n) slow

    def test_pad_toy_example(self, n: int = 1) -> None:
        self.fuzzing_test_network(
            abs_toy_pad_net, [0.001, 0.01], n, zono=True, dpf=True
        )
        self.fuzzing_test_network(abs_toy_pad_tiny_net, 0.01, n, zono=True, dpf=True)

    def test_toy_net(self, n: int = 1) -> None:
        self.fuzzing_test_network(toy_net, [0.001, 0.01], n, zono=True, dpf=True)

    # def test_toy_net_split(self, n: int = 1) -> None:
    #     splitting_dict = {
    #         "initial_splits": 2,
    #         "initial_split_dims": [0, 1],
    #         "max_depth": 5,
    #         "domain": AbstractDomain("DPF"),
    #         "batch_size": 100,
    #         "split_factor": 3,
    #     }
    #     self.fuzzing_test_network(
    #         toy_net,
    #         [0.001, 0.01],
    #         n,
    #         milp=False,
    #         dp=False,
    #         input_splitting=True,
    #         splitting_dict=splitting_dict,
    #     )
    #
    # def test_all_layer_split(self, n: int = 1) -> None:
    #     splitting_dict = {
    #         "initial_splits": 2,
    #         "initial_split_dims": [0, 1],
    #         "max_depth": 5,
    #         "domain": AbstractDomain("DPF"),
    #         "batch_size": 100,
    #         "split_factor": 3,
    #     }
    #     for _ in range(n):
    #         self.fuzzing_test_network(
    #             toy_all_layer_net_1d,
    #             [0.001, 0.01],
    #             1,
    #             milp=False,
    #             dp=False,
    #             input_splitting=True,
    #             splitting_dict=splitting_dict,
    #         )
    #
    #     splitting_dict = {
    #         "initial_splits": 1,
    #         "initial_split_dims": [0, 1],
    #         "max_depth": 5,
    #         "domain": AbstractDomain("dp"),
    #         "batch_size": 100,
    #         "split_factor": 3,
    #     }
    #
    #     for _ in range(n):
    #         self.fuzzing_test_network(
    #             toy_all_layer_net_1d,
    #             [0.001, 0.01],
    #             1,
    #             milp=False,
    #             dp=False,
    #             input_splitting=True,
    #             splitting_dict=splitting_dict,
    #         )
    #
    #     splitting_dict = {
    #         "initial_splits": 3,
    #         "initial_split_dims": [0],
    #         "max_depth": 5,
    #         "domain": AbstractDomain("zono"),
    #         "batch_size": 100,
    #         "split_factor": 3,
    #     }
    #     for _ in range(n):
    #         self.fuzzing_test_network(
    #             toy_all_layer_net_1d,
    #             [0.001, 0.01],
    #             1,
    #             milp=False,
    #             dp=False,
    #             input_splitting=True,
    #             splitting_dict=splitting_dict,
    #         )
    #
    #     splitting_dict = {
    #         "initial_splits": 3,
    #         "initial_split_dims": [0],
    #         "max_depth": 5,
    #         "domain": AbstractDomain("box"),
    #         "batch_size": 100,
    #         "split_factor": 3,
    #     }
    #     for _ in range(n):
    #         self.fuzzing_test_network(
    #             toy_all_layer_net_1d,
    #             [0.001, 0.01],
    #             1,
    #             milp=False,
    #             dp=False,
    #             input_splitting=True,
    #             splitting_dict=splitting_dict,
    #         )

    def test_toy_sig_net(self, n: int = 1) -> None:
        self.fuzzing_test_network(
            toy_sig_net, [0.001, 0.01], n, milp=False, zono=False, dpf=True
        )

    def test_toy_sig_tanh_net(self, n: int = 1) -> None:
        self.fuzzing_test_network(
            toy_sig_tanh_net, [0.001, 0.01], n, milp=False, zono=False, dpf=False
        )

    def test_toy_mixed(self, n: int = 1) -> None:
        self.fuzzing_test_network(
            toy_all_layer_net, [0.001, 0.01], n, zono=True, dpf=False
        )

    def test_toy_avg_pool_net(self, n: int = 1) -> None:
        self.fuzzing_test_network(
            toy_avg_pool_net, [0.001, 0.01], n, zono=True, dpf=True
        )

    def test_nn4sys(self) -> None:
        # Pass here as they are quite slow and test only sub-behaviour checked in other tests.
        pass
        # for _ in range(5):
        #     self.fuzzing_test_network(
        #         get_toy_split_block,
        #         0.5,
        #         1,
        #         milp=False,
        #         dpf=True,
        #     )
        # self.fuzzing_test_network(
        #     get_nn4sys_128d_splitblock,
        #     0.2,
        #     5,
        #     milp=False,
        #     dpf=True,
        # )
        # self.fuzzing_test_network(
        #     get_nn4sys_128d_block,
        #     0.2,
        #     5,
        #     milp=False,
        #     dpf=True,
        # )
        # self.fuzzing_test_network(
        #     get_nn4sys_128d_multipath_block_stacked,
        #     0.2,
        #     5,
        #     milp=False,
        #     dpf=True,
        # )

    def test_convtranspose2d_net(self, n: int = 1) -> None:

        self.fuzzing_test_network(
            toy_convtranspose2d_net, [0.001, 0.01], n, milp=False, zono=True, dpf=True
        )
        self.fuzzing_test_network(
            get_convtranspose2d_conv_net,
            [0.001, 0.01],
            n,
            milp=False,
            zono=True,
            dpf=True,
        )


if __name__ == "__main__":
    T = TestFuzzing()
    n = 50
    # T.test_toy_net_split(n)
    # T.test_all_layer_split(n)
    T.test_nn4sys()
    # T.test_convtranspose2d_net(n)
    T.test_maxpool_toy_example(n)
    T.test_pad_toy_example(n)
    T.test_toy_net(n)
    T.test_toy_sig_net(n)
    T.test_toy_sig_tanh_net(n)
    T.test_toy_mixed(n)
    T.test_toy_avg_pool_net(n)
