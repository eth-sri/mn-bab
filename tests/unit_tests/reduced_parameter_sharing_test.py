import torch

from src.utilities.config import make_optimizer_config
from tests.test_util import optimize_output_node_bounds_with_prima_crown, toy_net


class TestReducedParameterSharing:
    def test_toy_net(self) -> None:
        network = toy_net()[0]

        input_lb = torch.tensor([-1.0, -1.0]).unsqueeze(0)
        input_ub = torch.tensor([1.0, 1.0]).unsqueeze(0)

        optimizer_config = make_optimizer_config(
            optimize_alpha=True,
            optimize_prima=True,
            parameter_sharing={
                "fully_connected": "none",
                "conv2d": "in_channel",  # (test parsing, no conv2d layers in network)
            },
            parameter_sharing_layer_id_filter="layer_ids[-2:]",
        )

        assert optimizer_config.parameter_sharing_config.reduce_parameter_sharing

        (
            output_lb_with_alpha_prima_rps,
            output_ub_with_alpha_prima_rps,
        ) = optimize_output_node_bounds_with_prima_crown(
            network,
            0,
            input_lb,
            input_ub,
            custom_optimizer_config=optimizer_config,
        )

        assert output_lb_with_alpha_prima_rps >= 1.0
        assert output_ub_with_alpha_prima_rps <= 3.65402


if __name__ == "__main__":
    T = TestReducedParameterSharing()
    # torch.autograd.set_detect_anomaly(True)
    T.test_toy_net()
