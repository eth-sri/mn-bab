import torch

from tests.test_util import toy_net


class TestSequential:
    def test_set_layer_bounds_via_interval_propagation(self) -> None:
        """
        Correct bounds found by hand on toy_net.
        """
        model = toy_net()[0]

        input_lb = torch.tensor([-1.0, -1.0]).unsqueeze(0)
        input_ub = torch.tensor([1.0, 1.0]).unsqueeze(0)

        model.set_layer_bounds_via_interval_propagation(input_lb, input_ub)

        layer1_lb, layer1_ub = model.layers[0].input_bounds
        assert (layer1_lb == torch.tensor([-1, -1])).all()
        assert (layer1_ub == torch.tensor([1, 1])).all()

        layer2_lb, layer2_ub = model.layers[1].input_bounds
        assert (layer2_lb == torch.tensor([-2, -2])).all()
        assert (layer2_ub == torch.tensor([2, 2])).all()

        layer3_lb, layer3_ub = model.layers[2].input_bounds
        assert (layer3_lb == torch.tensor([0, 0])).all()
        assert (layer3_ub == torch.tensor([2, 2])).all()

        layer4_lb, layer4_ub = model.layers[3].input_bounds
        assert (layer4_lb == torch.tensor([0, -2])).all()
        assert (layer4_ub == torch.tensor([4, 2])).all()

        layer5_lb, layer5_ub = model.layers[4].input_bounds
        assert (layer5_lb == torch.tensor([0, 0])).all()
        assert (layer5_ub == torch.tensor([4, 2])).all()

        layer6_lb, layer6_ub = model.layers[5].input_bounds
        assert (layer6_lb == torch.tensor([1, 0])).all()
        assert (layer6_ub == torch.tensor([7, 2])).all()
