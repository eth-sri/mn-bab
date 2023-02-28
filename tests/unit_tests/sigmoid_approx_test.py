import numpy as np
import torch
import torch.nn as nn

from src.abstract_layers.abstract_network import AbstractNetwork
from src.abstract_layers.abstract_sigmoid import Sigmoid
from src.mn_bab_shape import AffineForm, MN_BaB_Shape
from src.state.tags import query_tag
from src.utilities.argument_parsing import get_config_from_json
from src.utilities.attacks import torch_whitebox_attack
from src.utilities.initialization import seed_everything
from src.utilities.loading.network import freeze_network, load_net_from
from tests.test_util import (
    get_deep_poly_bounds,
    optimize_output_node_bounds_with_prima_crown,
    toy_sig_net,
    toy_sig_tanh_net,
)


class TestSigmoid:
    """
    We test our MILP implementation for layer bounds and network verification
    """

    def test_sigmoid_bounds(self) -> None:
        seed_everything(42)
        shape = (100,)
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        sig = Sigmoid.from_concrete_module(nn.Sigmoid(), shape)
        sig = sig.to(device)
        sig.eval()
        for lb in torch.linspace(-5, 5, 100):
            lbs = lb * torch.ones(shape).to(device)
            ub_eps = torch.linspace(0, 10, 100).to(device)
            dummy_as = MN_BaB_Shape(
                query_id=query_tag(sig),
                query_prev_layer=None,
                queries_to_compute=None,
                lb=AffineForm(torch.tensor([[[0]]], device=device)),
                ub=AffineForm(torch.tensor([[[0]]], device=device)),
                unstable_queries=None,
                subproblem_state=None,
            )
            ubs = lbs + ub_eps
            (
                lb_slope,
                ub_slope,
                lb_intercept,
                ub_intercept,
            ) = sig.get_approximation_slopes_and_intercepts(
                bounds=(lbs, ubs), abstract_shape=dummy_as
            )
            # Check if bounds are valid
            lb_dist = []
            ub_dist = []
            for i in torch.linspace(0, 1, 100):
                check_x = lb + i * ub_eps
                lb_dist.append(
                    torch.mean(
                        (torch.sigmoid(check_x) - (lb_slope * check_x + lb_intercept))
                    ).item()
                )
                ub_dist.append(
                    torch.mean(
                        ((ub_slope * check_x + ub_intercept) - torch.sigmoid(check_x))
                    ).item()
                )
                assert (
                    lb_slope * check_x + lb_intercept <= torch.sigmoid(check_x)
                ).all(), "Lower bound failure"
                assert (
                    ub_slope * check_x + ub_intercept >= torch.sigmoid(check_x)
                ).all(), "Upper bound failure"

            print(
                f"Mean lb-dist = {np.mean(np.array(lb_dist))} Max lb-dist = {np.max(np.array(lb_dist))}"
            )
            print(
                f"Mean ub-dist = {np.mean(np.array(ub_dist))} Max ub-dist = {np.max(np.array(ub_dist))}"
            )

    def test_sigmoid_net_sound(self) -> None:
        seed_everything(42)
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        network = toy_sig_net()[0]
        network = network.to(device)
        network.eval()
        input_lb = torch.tensor([-1.0, -1.0]).unsqueeze(0).to(device)
        input_ub = torch.tensor([1.0, 1.0]).unsqueeze(0).to(device)

        (
            output_lb_without_alpha,
            output_ub_without_alpha,
        ) = get_deep_poly_bounds(network, input_lb, input_ub)

        for x1 in torch.linspace(float(input_lb[0][0]), float(input_ub[0][0]), 50):
            for x2 in torch.linspace(float(input_lb[0][1]), float(input_ub[0][1]), 50):
                out = network(torch.tensor([x1, x2], device=device))
                assert out <= output_ub_without_alpha
                assert out >= output_lb_without_alpha

    def test_sigmoid_bound_optimization(self) -> None:
        seed_everything(42)
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        network = toy_sig_net()[0]
        network = network.to(device)
        network.eval()
        freeze_network(network)
        input_lb = torch.tensor([-1.0, -1.0]).unsqueeze(0).to(device)
        input_ub = torch.tensor([1.0, 1.0]).unsqueeze(0).to(device)

        (
            output_lb_without_alpha,
            output_ub_without_alpha,
        ) = get_deep_poly_bounds(network, input_lb, input_ub)

        (
            output_lb_with_alpha,
            output_ub_with_alpha,
        ) = optimize_output_node_bounds_with_prima_crown(
            network, 0, input_lb, input_ub, optimize_alpha=True, optimize_prima=False
        )

        assert output_lb_with_alpha >= output_lb_without_alpha
        assert output_ub_with_alpha <= output_ub_without_alpha

    def test_sigmoid_tanh_layers(self) -> None:
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        network = toy_sig_tanh_net()[0]
        network = network.to(device)
        network.eval()
        freeze_network(network)
        input_lb = torch.tensor([-1.0, -1.0]).unsqueeze(0).to(device)
        input_ub = torch.tensor([1.0, 1.0]).unsqueeze(0).to(device)

        (
            output_lb_without_alpha,
            output_ub_without_alpha,
        ) = get_deep_poly_bounds(network, input_lb, input_ub)

        (
            output_lb_with_alpha,
            output_ub_with_alpha,
        ) = optimize_output_node_bounds_with_prima_crown(
            network, 0, input_lb, input_ub, optimize_alpha=True, optimize_prima=False
        )

        assert output_lb_with_alpha >= output_lb_without_alpha
        assert output_ub_with_alpha <= output_ub_without_alpha

    def test_sigmoid_large_net(self) -> None:
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        config = get_config_from_json("configs/baseline/mnist_sig_2_50.json")
        seed_everything(config.random_seed)
        network = load_net_from(config)
        network = network.to(device)
        network.eval()
        assert isinstance(network, nn.Sequential)
        network = AbstractNetwork.from_concrete_module(network, config.input_dim)
        freeze_network(network)
        eps = 0.1
        for i in range(5):
            input = torch.rand(config.input_dim).unsqueeze(0).to(device)
            input_lb = torch.clamp(input - eps, min=0)
            input_ub = torch.clamp(input + eps, max=1)
            (
                output_lb_without_alpha,
                output_ub_without_alpha,
            ) = get_deep_poly_bounds(network, input_lb, input_ub)

            i = int(torch.randint(0, 10, (1,)).item())
            (
                output_lb_with_alpha,
                output_ub_with_alpha,
            ) = optimize_output_node_bounds_with_prima_crown(
                network,
                i,
                input_lb,
                input_ub,
                optimize_alpha=True,
                optimize_prima=False,
            )
            assert output_lb_with_alpha > output_lb_without_alpha[0][i]
            assert output_ub_with_alpha < output_ub_without_alpha[0][i]

    def test_sigmoid_net_large_soundness(self) -> None:
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        dtype = torch.get_default_dtype()
        config = get_config_from_json("configs/baseline/mnist_sig_2_50.json")
        seed_everything(config.random_seed)
        network = load_net_from(config)
        network = network.to(device).to(dtype)
        network.eval()
        assert isinstance(network, nn.Sequential)
        network = AbstractNetwork.from_concrete_module(network, config.input_dim)
        freeze_network(network)
        eps = 0.1
        for i in range(5):
            network.reset_output_bounds()
            network.reset_input_bounds()

            input = torch.rand(config.input_dim).unsqueeze(0).to(device)
            input_lb = torch.clamp(input - eps, min=0)
            input_ub = torch.clamp(input + eps, max=1)
            (
                output_lb_without_alpha,
                output_ub_without_alpha,
            ) = get_deep_poly_bounds(network, input_lb, input_ub)

            i = int(torch.randint(0, 10, (1,)).item())
            properties_to_verify = [[(i, j, 0)] for j in range(10) if j != i]
            adversarial_example, worst_x = torch_whitebox_attack(
                network,
                input_lb.device,
                input,
                properties_to_verify,
                input_lb,
                input_ub,
                restarts=5,
            )

            (
                output_lb_with_alpha,
                output_ub_with_alpha,
            ) = optimize_output_node_bounds_with_prima_crown(
                network,
                i,
                input_lb,
                input_ub,
                optimize_alpha=True,
                optimize_prima=False,
            )

            assert output_lb_with_alpha > output_lb_without_alpha[0][i]
            assert output_ub_with_alpha < output_ub_without_alpha[0][i]

            x = input
            adv_out = network(x)
            assert output_lb_with_alpha < adv_out[0][i]
            assert output_ub_with_alpha > adv_out[0][i]

            if worst_x is not None:
                x = torch.tensor(worst_x[0], device=device)
                adv_out = network(x)
                assert output_lb_with_alpha < adv_out[i]
                assert output_ub_with_alpha > adv_out[i]
            assert adversarial_example is not None
            for x_ndarray in adversarial_example:
                x = torch.tensor(x_ndarray, device=device)[0]
                adv_out = network(x)
                assert output_lb_with_alpha < adv_out[i]
                assert output_ub_with_alpha > adv_out[i]

            b_s = 50
            rand_sample_batch = (
                torch.rand((b_s, *config.input_dim), device=device) * eps
            )
            rand_sample_batch = torch.clamp(
                input.repeat((b_s, 1)) + rand_sample_batch, 0, 1
            )
            rand_out = network(rand_sample_batch)
            for j in range(b_s):
                assert output_lb_with_alpha < rand_out[j][i]
                assert output_ub_with_alpha > rand_out[j][i]

            rand_sample_batch = (
                torch.randint(-1, 1, (b_s, *config.input_dim), device=device) * eps
            )
            rand_sample_batch = torch.clamp(
                input.repeat((b_s, 1)) + rand_sample_batch, 0, 1
            )
            rand_out = network(rand_sample_batch)
            for j in range(b_s):
                assert output_lb_with_alpha < rand_out[j][i]
                assert output_ub_with_alpha > rand_out[j][i]

    def test_large_sigmoid_input_bounds(self) -> None:
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        seed_everything(42)
        input_lb = torch.tensor(
            [
                -1000,
                -1000,
                -1000,
                -1000,
                -500,
                -500,
                -500,
                -500,
                -499,
                -499,
                -499,
                -499,
            ],
            device=device,
        )
        input_ub = torch.tensor(
            [-1000, -500, 499, 500, -1000, -500, 499, 500, -1000, -500, 499, 500],
            device=device,
        )
        sig_layer = Sigmoid(input_lb.shape)
        sig_layer = sig_layer.to(device)
        sig_layer.eval()

        (
            lb_slope,
            ub_slope,
            lb_intercept,
            ub_intercept,
        ) = sig_layer.get_approximation_slopes_and_intercepts((input_lb, input_ub))

        for i, (lb, ub) in enumerate(zip(input_lb, input_ub)):
            if lb < 0:
                assert abs(lb_slope[i] - 0) <= 1e-6
                assert abs(lb_intercept[i] - 0) <= 1e-6
            if ub > 0:
                assert abs(ub_slope[i] - 0) <= 1e-6
                assert abs(ub_intercept[i] - 1) <= 1e-6

        print("Done")


if __name__ == "__main__":
    t = TestSigmoid()
    t.test_sigmoid_net_sound()
    # t.test_sigmoid_net_large_soundness()
    # t.test_sigmoid_large_net()
    # t.test_large_sigmoid_input_bounds()
    # t.test_sigmoid_bounds()
    t.test_sigmoid_bound_optimization()
    t.test_sigmoid_tanh_layers()
