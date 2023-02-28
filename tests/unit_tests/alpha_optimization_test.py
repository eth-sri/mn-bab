import torch

from tests.test_util import (
    get_deep_poly_bounds,
    optimize_output_node_bounds_with_prima_crown,
    toy_net,
)


class TestAlphaOptimization:
    def test_toy_net(self) -> None:
        network = toy_net()[0]

        input_lb = torch.tensor([-1.0, -1.0]).unsqueeze(0)
        input_ub = torch.tensor([1.0, 1.0]).unsqueeze(0)

        (
            output_lb_without_alpha,
            output_ub_without_alpha,
        ) = get_deep_poly_bounds(network, input_lb, input_ub)

        (
            output_lb_with_alpha,
            output_ub_with_alpha,
        ) = optimize_output_node_bounds_with_prima_crown(
            network, 0, input_lb, input_ub, optimize_alpha=True
        )

        assert output_lb_with_alpha >= output_lb_without_alpha
        assert output_ub_with_alpha <= output_ub_without_alpha


if __name__ == "__main__":
    T = TestAlphaOptimization()
    T.test_toy_net()
