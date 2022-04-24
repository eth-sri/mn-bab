import torch
from torch import Tensor

from src.abstract_layers.abstract_linear import Linear
from src.mn_bab_shape import MN_BaB_Shape


class TestLinear:
    def test_backsubstitution_mn_bab(self) -> None:
        lb_coef = torch.eye(2).unsqueeze(0)
        ub_coef = 2 * lb_coef
        initial_shape = MN_BaB_Shape(lb_coef, ub_coef)
        assert isinstance(initial_shape.lb_coef, Tensor)
        assert isinstance(initial_shape.ub_coef, Tensor)

        layer = Linear(10, 2)

        expected_lb_coef = initial_shape.lb_coef.matmul(layer.weight)
        expected_ub_coef = initial_shape.ub_coef.matmul(layer.weight)

        expected_lb_bias = initial_shape.lb_coef.matmul(layer.bias)
        expected_ub_bias = initial_shape.ub_coef.matmul(layer.bias)

        expected_shape = MN_BaB_Shape(
            expected_lb_coef, expected_ub_coef, expected_lb_bias, expected_ub_bias
        )
        assert isinstance(expected_shape.lb_coef, Tensor)
        assert isinstance(expected_shape.ub_coef, Tensor)

        actual_shape = layer.backsubstitute(initial_shape)

        assert expected_shape.lb_coef.equal(actual_shape.lb_coef)
        assert expected_shape.ub_coef.equal(actual_shape.ub_coef)
        assert expected_shape.lb_bias.equal(actual_shape.lb_bias)
        assert expected_shape.ub_bias.equal(actual_shape.ub_bias)

    def test_propagte_interval_identity_layer(self) -> None:
        layer = Linear(2, 2)
        layer.weight.data = torch.eye(2)
        layer.bias.data = torch.zeros(2)

        input_lb = torch.tensor([-1.0, -1.0])
        input_ub = torch.tensor([1.0, 1.0])

        output_lb, output_ub = layer.propagate_interval((input_lb, input_ub))

        assert (output_lb == input_lb).all()
        assert (output_ub == input_ub).all()

    def test_propagate_interval_toy_example_layer(self) -> None:
        layer = Linear(2, 2)
        layer.weight.data = torch.tensor([[1.0, 1.0], [1.0, -1.0]])
        layer.bias.data = torch.tensor([1, -1])

        input_lb = torch.tensor([-1.0, -1.0])
        input_ub = torch.tensor([1.0, 1.0])

        expected_output_lb = torch.tensor([-1, -3])
        expected_output_ub = torch.tensor([3, 1])
        output_lb, output_ub = layer.propagate_interval((input_lb, input_ub))

        assert (output_lb == expected_output_lb).all()
        assert (output_ub == expected_output_ub).all()
