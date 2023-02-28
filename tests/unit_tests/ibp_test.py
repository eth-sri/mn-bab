from typing import Dict, Tuple

import torch
from torch import Tensor

from src.milp_network import MILPNetwork
from tests.test_util import get_deep_poly_bounds, toy_net


class TestIBP:
    """
    We test our IBP implementation (also with respect to existing bound reusage)
    """

    def test_milp_toy_example(self) -> None:
        network = toy_net()[0]
        input_lb = torch.tensor([[-1.0, -1.0]])
        input_ub = torch.tensor([[1.0, 1.0]])
        # Our new implementation
        input_lb = input_lb.unsqueeze(0)
        input_ub = input_ub.unsqueeze(0)

        # Simple IBP pass
        has_intermediate_layer_bounds = MILPNetwork._check_layer_bounds(network.layers)
        assert not has_intermediate_layer_bounds
        output_lb, output_ub = network.set_layer_bounds_via_interval_propagation(
            input_lb, input_ub, use_existing_bounds=False
        )
        print("Basic IBP")
        prior_bounds: Dict[int, Tuple[Tensor]] = {}
        for i, layer in enumerate(network.layers):
            if hasattr(layer, "input_bounds"):
                print(f"Layer: {i} Bounds: {layer.input_bounds}")
                prior_bounds[i] = layer.input_bounds

        # Reset bounds
        network.reset_input_bounds()
        network.reset_output_bounds()

        # Run DP pass yielding better bounds
        get_deep_poly_bounds(network, input_lb[0], input_ub[0])
        has_intermediate_layer_bounds = MILPNetwork._check_layer_bounds(network.layers)
        assert not has_intermediate_layer_bounds
        print("After DP")
        for i, layer in enumerate(network.layers):
            if hasattr(layer, "input_bounds"):
                print(f"Layer: {i} Bounds: {layer.input_bounds}")

        # Run IBP re-using intermediate results
        output_lb, output_ub = network.set_layer_bounds_via_interval_propagation(
            input_lb, input_ub, use_existing_bounds=True
        )

        print("IBP Reusing DP Bounds")
        for i, layer in enumerate(network.layers):
            if hasattr(layer, "input_bounds"):
                print(f"Layer: {i} Bounds: {layer.input_bounds}")
                assert (layer.input_bounds[0] >= prior_bounds[i][0]).all()
                assert (layer.input_bounds[1] <= prior_bounds[i][1]).all()  # type: ignore[misc] # mypy throws false index out of range


if __name__ == "__main__":
    T = TestIBP()
    T.test_milp_toy_example()
