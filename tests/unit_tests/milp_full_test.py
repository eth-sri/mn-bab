import random
import time

import torch
from torch import nn

from src.abstract_layers.abstract_network import AbstractNetwork
from src.milp_network import MILPNetwork
from src.state.tags import layer_tag
from src.utilities.argument_parsing import get_config_from_json
from src.utilities.initialization import seed_everything
from src.utilities.loading.network import load_net_from
from tests.test_util import get_deep_poly_bounds, toy_all_layer_net, toy_net


class TestMILP:
    """
    We test our MILP implementation for layer bounds and network verification
    """

    def test_milp_toy_example(self) -> None:
        network = toy_net()[0]
        input_lb = torch.tensor([[-1.0, -1.0]])
        input_ub = torch.tensor([[1.0, 1.0]])
        # Our new implementation
        # input_lb = input_lb.unsqueeze(0)
        # input_ub = input_ub.unsqueeze(0)
        x = (input_lb + input_ub) / 2
        milp_model = MILPNetwork.build_model_from_abstract_net(
            x, input_lb, input_ub, network
        )
        lbs, ubs = milp_model.get_network_output_bounds()
        assert lbs[0] == 1
        assert ubs[0] == 3

    def test_milp_all_layers(self) -> None:
        torch.manual_seed(0)
        random.seed(0)
        eps = 0.1
        num_eps = 1e-6
        for i in range(10):
            network = toy_all_layer_net()[0]
            input = torch.rand(size=(1, 1, 5, 5))
            # input = torch.rand(size=(1, 27))
            input_lb, input_ub = input - eps, input + eps
            q = network(input)
            milp_model = MILPNetwork.build_model_from_abstract_net(
                input, input_lb, input_ub, network
            )
            lbs, ubs = milp_model.get_network_output_bounds()
            (
                output_lb_deep_poly,
                output_ub_deep_poly,
            ) = get_deep_poly_bounds(network, input_lb, input_ub)
            assert (q >= lbs).all()
            assert (q <= ubs).all()
            assert (
                output_lb_deep_poly < lbs + num_eps
            ).all(), (
                f"Found violation lb {torch.max(output_lb_deep_poly - (lbs + num_eps))}"
            )
            if not (output_ub_deep_poly > ubs - num_eps).all():
                for i, layer in enumerate(milp_model.net.layers):
                    lbi, ubi = milp_model.get_network_bounds_at_layer_multi(
                        layer_tag(layer),
                        timeout_per_instance=20,
                        timeout_total=400,
                        timeout=time.time() + 400,
                    )
                    print(f"Layer {i} ============== \n LBS: {lbi} \n UBS: {ubi}")
                assert (
                    output_ub_deep_poly > ubs - num_eps
                ).all(), f"Found violation ub {torch.max(ubs - num_eps - output_ub_deep_poly)}"

    def test_recursive_encoding(self) -> None:
        config = get_config_from_json("configs/baseline/cifar10_resnet_4b_bn.json")
        seed_everything(config.random_seed)
        network = load_net_from(config)
        assert isinstance(network, nn.Sequential)
        abs_net = AbstractNetwork.from_concrete_module(network, config.input_dim)

        eps = 0.1
        input = torch.rand(config.input_dim).unsqueeze(0)
        input_lb = torch.clamp(input - eps, min=0)
        input_ub = torch.clamp(input + eps, max=1)
        milp_model = MILPNetwork.build_model_from_abstract_net(
            input, input_lb, input_ub, abs_net
        )

        net_layers = dict(network.named_modules()).items()
        filtered_net_layers = set([k for (k, v) in net_layers])
        milp_layers = set(milp_model.layer_id_to_prefix_map.values())
        # The outer ResNet identifier and 8 individual path identifiers
        assert len(filtered_net_layers - milp_layers) == 9

    def test_intermediate_layer_access(self) -> None:
        network = toy_net()[0]
        input_lb = torch.tensor([[-1.0, -1.0]])
        input_ub = torch.tensor([[1.0, 1.0]])
        # Our new implementation
        # input_lb = input_lb.unsqueeze(0)
        # input_ub = input_ub.unsqueeze(0)
        x = (input_lb + input_ub) / 2
        milp_model = MILPNetwork.build_model_from_abstract_net(
            x, input_lb, input_ub, network
        )
        assert len(milp_model.layer_id_to_prefix_map) == 6

    def test_intermediate_refinement_single(self) -> None:
        network = toy_net()[0]
        input_lb = torch.tensor([[-1.0, -1.0]])
        input_ub = torch.tensor([[1.0, 1.0]])
        # Our new implementation
        input_lb = input_lb.unsqueeze(0)
        input_ub = input_ub.unsqueeze(0)
        x = (input_lb + input_ub) / 2
        milp_model = MILPNetwork.build_model_from_abstract_net(
            x, input_lb, input_ub, network, max_milp_neurons=1
        )
        for i, layer in enumerate(milp_model.net.layers):
            lbs, ubs = milp_model._get_network_bounds_at_layer_single(
                layer_tag(layer), timeout=20
            )
            print(f"Layer {i} - LBS: {lbs} UBS: {ubs}")
            assert (lbs >= milp_model.net.layers[i].output_bounds[0]).all()
            assert (ubs <= milp_model.net.layers[i].output_bounds[1]).all()

    def test_intermediate_refinement_multi(self) -> None:
        network = toy_net()[0]
        input_lb = torch.tensor([[-1.0, -1.0]])
        input_ub = torch.tensor([[1.0, 1.0]])
        # Our new implementation
        input_lb = input_lb.unsqueeze(0)
        input_ub = input_ub.unsqueeze(0)
        x = (input_lb + input_ub) / 2
        milp_model = MILPNetwork.build_model_from_abstract_net(
            x, input_lb, input_ub, network
        )
        for i, layer in enumerate(milp_model.net.layers):
            lbs, ubs = milp_model.get_network_bounds_at_layer_multi(
                layer_tag(layer),
                timeout_per_instance=20,
                timeout_total=400,
                timeout=time.time() + 400,
            )
            print(f"Layer {i} - LBS: {lbs} UBS: {ubs}")
            assert (lbs >= milp_model.net.layers[i].output_bounds[0]).all()
            assert (ubs <= milp_model.net.layers[i].output_bounds[1]).all()


if __name__ == "__main__":
    T = TestMILP()
    # T.test_intermediate_refinement_single()
    T.test_milp_toy_example()
