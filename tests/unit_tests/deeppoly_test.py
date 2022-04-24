import torch

from tests.test_util import get_deep_poly_bounds, toy_net

# DeepPoly paper source: https://files.sri.inf.ethz.ch/website/papers/DeepPoly.pdf


class TestDeepPoly:
    """
    PRIMA_CROWN is an extension of DeepPoly. Without any optimization,
    it should yield the same approximations as DeepPoly does.
    """

    def test_toy_net(self) -> None:
        """
        Expected lower and upper bound taken from DeepPoly paper
        """
        expected_output_lb = 1
        expected_output_ub = 4

        model = toy_net()

        input_lb = torch.tensor([-1, -1])
        input_ub = torch.tensor([1, 1])

        output_lb, output_ub = get_deep_poly_bounds(model, input_lb, input_ub)

        assert output_lb == expected_output_lb
        assert output_ub == expected_output_ub
