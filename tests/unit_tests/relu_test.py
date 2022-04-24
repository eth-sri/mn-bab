import pytest
import torch
from torch import Tensor

from src.abstract_layers.abstract_relu import ReLU
from src.mn_bab_shape import MN_BaB_Shape


class TestReLU:
    def test_backsubstitution_with_missing_bounds(self) -> None:
        with pytest.raises(RuntimeError):
            dummy_shape = MN_BaB_Shape.construct_to_bound_all_outputs(
                torch.device("cpu"), (2,)
            )
            layer = ReLU((1,))
            layer.backsubstitute(dummy_shape)

    def test_approximation_stable_inactive(self) -> None:
        layer_lb = torch.full(size=(1, 2), fill_value=-2)
        layer_ub = torch.full(size=(1, 2), fill_value=-1)
        layer = ReLU((2,))
        layer.update_input_bounds((layer_lb, layer_ub))

        assert layer.input_bounds  # mypy
        lb_slope = layer._get_lower_approximation_slopes(layer.input_bounds)
        ub_slope = layer._get_upper_approximation_slopes(layer.input_bounds)
        lb_intercept, ub_intercept = layer._get_approximation_intercepts(
            layer.input_bounds
        )

        assert (lb_slope == 0).all()
        assert (ub_slope == 0).all()
        assert (lb_intercept == 0).all()
        assert (ub_intercept == 0).all()

    def test_approximation_stable_active(self) -> None:
        layer_lb = torch.full(size=(1, 2), fill_value=1)
        layer_ub = torch.full(size=(1, 2), fill_value=2)
        layer = ReLU((2,))
        layer.update_input_bounds((layer_lb, layer_ub))

        assert layer.input_bounds  # mypy
        lb_slope = layer._get_lower_approximation_slopes(layer.input_bounds)
        ub_slope = layer._get_upper_approximation_slopes(layer.input_bounds)
        lb_intercept, ub_intercept = layer._get_approximation_intercepts(
            layer.input_bounds
        )

        assert (lb_slope == 1).all()
        assert (ub_slope == 1).all()
        assert (lb_intercept == 0).all()
        assert (ub_intercept == 0).all()

    def test_approximation_unstable_lower_triangle(self) -> None:
        layer_lb = torch.full(size=(1, 2), fill_value=-1)
        layer_ub = torch.full(size=(1, 2), fill_value=2)
        layer = ReLU((2,))
        layer.update_input_bounds((layer_lb, layer_ub))

        expected_ub_slope = 2 / (2 - (-1))
        expected_ub_intercept = -(-1) * expected_ub_slope

        assert layer.input_bounds  # mypy
        lb_slope = layer._get_lower_approximation_slopes(layer.input_bounds)
        ub_slope = layer._get_upper_approximation_slopes(layer.input_bounds)
        lb_intercept, ub_intercept = layer._get_approximation_intercepts(
            layer.input_bounds
        )

        assert (lb_slope == 1).all()
        assert (ub_slope == expected_ub_slope).all()
        assert (lb_intercept == 0).all()
        assert (ub_intercept == expected_ub_intercept).all()

    def test_approximation_unstable_upper_triangle(self) -> None:
        layer_lb = torch.full(size=(1, 2), fill_value=-2)
        layer_ub = torch.full(size=(1, 2), fill_value=1)
        layer = ReLU((2,))
        layer.update_input_bounds((layer_lb, layer_ub))

        expected_ub_slope = 1 / (1 - (-2))
        expected_ub_intercept = -(-2) * expected_ub_slope

        assert layer.input_bounds  # mypy
        lb_slope = layer._get_lower_approximation_slopes(layer.input_bounds)
        ub_slope = layer._get_upper_approximation_slopes(layer.input_bounds)
        lb_intercept, ub_intercept = layer._get_approximation_intercepts(
            layer.input_bounds
        )

        assert (lb_slope == 0).all()
        assert (ub_slope == expected_ub_slope).all()
        assert (lb_intercept == 0).all()
        assert (ub_intercept == expected_ub_intercept).all()

    def test_mn_bab_backsubstitution_coef(self) -> None:
        lb_coef = torch.eye(2).unsqueeze(0)
        ub_coef = torch.tensor([[1.0, 0.0], [0.0, -1.0]]).unsqueeze(0)
        initial_shape = MN_BaB_Shape(lb_coef, ub_coef)

        layer_lb = torch.full(size=(1, 2), fill_value=-2.0)
        layer_ub = torch.full(size=(1, 2), fill_value=1.0)
        layer = ReLU((2,))
        layer.update_input_bounds((layer_lb, layer_ub))

        resulting_shape = layer.backsubstitute(initial_shape)

        expected_lb_slope = 0
        expected_ub_slope = 1 / (1 - (-2))

        assert (
            resulting_shape.lb_coef
            == torch.zeros(2, 2).fill_diagonal_(expected_lb_slope)
        ).all()
        assert isinstance(resulting_shape.ub_coef, Tensor)
        assert resulting_shape.ub_coef[0, 0, 0] == expected_ub_slope
        assert resulting_shape.ub_coef[0, 1, 1] == expected_lb_slope

    def test_mn_bab_backsubstitution_bias(self) -> None:
        lb_coef = torch.eye(2).unsqueeze(0)
        ub_coef = torch.tensor([[1.0, 0.0], [0.0, -1.0]]).unsqueeze(0)
        initial_shape = MN_BaB_Shape(lb_coef, ub_coef)

        layer_lb = torch.full(size=(1, 2), fill_value=-2.0)
        layer_ub = torch.full(size=(1, 2), fill_value=1.0)
        layer = ReLU((2,))
        layer.update_input_bounds((layer_lb, layer_ub))

        resulting_shape = layer.backsubstitute(initial_shape)

        expected_lb_intercept = 0
        expected_ub_slope = 1 / (1 - (-2))
        expected_ub_intercept = -(-2) * expected_ub_slope

        assert (resulting_shape.lb_bias == expected_lb_intercept).all()
        assert resulting_shape.ub_bias[0, 0] == expected_ub_intercept
        assert resulting_shape.ub_bias[0, 1] == expected_lb_intercept

    def test_propagate_interval_toy_example(self) -> None:
        layer = ReLU((2,))

        input_lb = torch.tensor([-2, -2])
        input_ub = torch.tensor([2, 2])

        expected_output_lb = torch.tensor([0, 0])
        expected_output_ub = torch.tensor([2, 2])
        output_lb, output_ub = layer.propagate_interval((input_lb, input_ub))

        assert (output_lb == expected_output_lb).all()
        assert (output_ub == expected_output_ub).all()

    def test_propagate_interval_forcing_lower_triangle_approximation(self) -> None:
        layer = ReLU((2,))

        input_lb = torch.tensor([-2, -2])
        input_ub = torch.tensor([2.1, 2.1])

        expected_output_lb = torch.tensor([0, 0])
        expected_output_ub = torch.tensor([2.1, 2.1])
        output_lb, output_ub = layer.propagate_interval((input_lb, input_ub))

        assert (output_lb == expected_output_lb).all()
        assert (output_ub == expected_output_ub).all()
