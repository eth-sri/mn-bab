import torch

from tests.test_util import optimize_output_node_bounds_with_prima_crown, toy_net


class TestPrimaOptimization:
    def test_toy_net(self) -> None:
        network = toy_net()[0]

        input_lb = torch.tensor([-1.0, -1.0]).unsqueeze(0)
        input_ub = torch.tensor([1.0, 1.0]).unsqueeze(0)

        (
            output_lb_with_alpha,
            output_ub_with_alpha,
        ) = optimize_output_node_bounds_with_prima_crown(
            network, 0, input_lb, input_ub, optimize_alpha=True
        )

        (
            output_lb_with_alpha_prima,
            output_ub_with_alpha_prima,
        ) = optimize_output_node_bounds_with_prima_crown(
            network, 0, input_lb, input_ub, optimize_alpha=True, optimize_prima=True
        )

        assert output_lb_with_alpha_prima >= output_lb_with_alpha
        assert output_ub_with_alpha_prima <= output_ub_with_alpha


if __name__ == "__main__":
    T = TestPrimaOptimization()
    # torch.autograd.set_detect_anomaly(True)
    T.test_toy_net()
