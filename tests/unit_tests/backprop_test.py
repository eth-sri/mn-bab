import torch

from tests.test_util import (
    get_deep_poly_bounds,
    optimize_output_node_bounds_with_prima_crown,
    toy_stack_seq_net,
)


class TestBackprop:
    """
    Test the more intricate parts of the backsubstitution implementation.
    """

    def test_seq_layer_stacking_sound(self) -> None:
        network = toy_stack_seq_net()

        print(network)

        input_lb = torch.tensor([0, -1.0]).unsqueeze(0)
        input_ub = torch.tensor([1.0, 1.0]).unsqueeze(0)

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
        assert output_ub_with_alpha < output_ub_without_alpha

        assert output_lb_without_alpha == 0.0
        assert torch.isclose(output_ub_without_alpha, torch.tensor(4 + 2 / 3))

        assert output_lb_with_alpha == 0.0
        assert output_ub_with_alpha == 4.0

        concrete_lb = float("inf")
        concrete_ub = -float("inf")

        for x1 in torch.linspace(float(input_lb[0][0]), float(input_ub[0][0]), 5):
            for x2 in torch.linspace(float(input_lb[0][1]), float(input_ub[0][1]), 5):
                out = network(torch.Tensor([x1, x2]))
                concrete_lb = min(concrete_lb, float(out))
                concrete_ub = max(concrete_ub, float(out))
        assert concrete_lb == output_lb_with_alpha
        assert concrete_ub == output_ub_with_alpha


if __name__ == "__main__":
    T = TestBackprop()
    T.test_seq_layer_stacking_sound()
