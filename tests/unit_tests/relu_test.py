import pytest
import torch
from torch import Tensor

from src.abstract_domains.DP_f import DeepPoly_f
from src.abstract_domains.zonotope import HybridZonotope
from src.abstract_layers.abstract_relu import ReLU
from src.mn_bab_shape import AffineForm, MN_BaB_Shape
from src.state.tags import query_tag
from src.utilities.config import make_backsubstitution_config
from src.utilities.queries import get_output_bound_initial_query_coef
from tests.test_util import set_torch_precision


class TestReLU:
    def test_backsubstitution_with_missing_bounds(self) -> None:
        with pytest.raises(RuntimeError):
            layer = ReLU((1,))
            query_coef = get_output_bound_initial_query_coef(
                dim=(2,),
                intermediate_bounds_to_recompute=None,  # get all
                use_dependence_sets=False,
                batch_size=1,
                device=torch.device("cpu"),
                dtype=None,  # TODO: should this be something else?
            )
            dummy_shape = MN_BaB_Shape(
                query_id=query_tag(layer),
                query_prev_layer=None,
                queries_to_compute=None,
                lb=AffineForm(query_coef),
                ub=AffineForm(query_coef),
                unstable_queries=None,
                subproblem_state=None,
            )
            layer.backsubstitute(make_backsubstitution_config(), dummy_shape)

    @set_torch_precision(torch.float32)
    def test_approximation_stable_inactive(self) -> None:
        layer = ReLU((2,))
        layer_lb = torch.full(size=(1, 2), fill_value=-2)
        layer_ub = torch.full(size=(1, 2), fill_value=-1)
        layer.update_input_bounds((layer_lb, layer_ub))

        assert layer.input_bounds  # mypy
        lb_slope = layer._get_lower_approximation_slopes(
            make_backsubstitution_config(), layer.input_bounds
        )
        ub_slope = layer._get_upper_approximation_slopes(
            make_backsubstitution_config(), layer.input_bounds
        )
        lb_intercept, ub_intercept = layer._get_approximation_intercepts(
            layer.input_bounds
        )

        assert (lb_slope == 0).all()
        assert (ub_slope == 0).all()
        assert (lb_intercept == 0).all()
        assert (ub_intercept == 0).all()

    @set_torch_precision(torch.float32)
    def test_approximation_stable_active(self) -> None:
        layer = ReLU((2,))
        layer_lb = torch.full(size=(1, 2), fill_value=1)
        layer_ub = torch.full(size=(1, 2), fill_value=2)
        layer.update_input_bounds((layer_lb, layer_ub))

        assert layer.input_bounds  # mypy
        lb_slope = layer._get_lower_approximation_slopes(
            make_backsubstitution_config(), layer.input_bounds
        )
        ub_slope = layer._get_upper_approximation_slopes(
            make_backsubstitution_config(), layer.input_bounds
        )
        lb_intercept, ub_intercept = layer._get_approximation_intercepts(
            layer.input_bounds
        )

        assert (lb_slope == 1).all()
        assert (ub_slope == 1).all()
        assert (lb_intercept == 0).all()
        assert (ub_intercept == 0).all()

    @set_torch_precision(torch.float32)
    def test_approximation_unstable_lower_triangle(self) -> None:
        layer = ReLU((2,))
        layer_lb = torch.full(size=(1, 2), fill_value=-1)
        layer_ub = torch.full(size=(1, 2), fill_value=2)
        layer.update_input_bounds((layer_lb, layer_ub))

        expected_ub_slope = 2 / (2 - (-1))
        expected_ub_intercept = -(-1) * expected_ub_slope

        assert layer.input_bounds  # mypy
        lb_slope = layer._get_lower_approximation_slopes(
            make_backsubstitution_config(), layer.input_bounds
        )
        ub_slope = layer._get_upper_approximation_slopes(
            make_backsubstitution_config(), layer.input_bounds
        )
        lb_intercept, ub_intercept = layer._get_approximation_intercepts(
            layer.input_bounds
        )

        assert (lb_slope == 1).all()
        assert (ub_slope == expected_ub_slope).all()
        assert (lb_intercept == 0).all()
        assert (ub_intercept == expected_ub_intercept).all()

    @set_torch_precision(torch.float32)
    def test_approximation_unstable_upper_triangle(self) -> None:
        layer = ReLU((2,))
        layer_lb = torch.full(size=(1, 2), fill_value=-2)
        layer_ub = torch.full(size=(1, 2), fill_value=1)
        layer.update_input_bounds((layer_lb, layer_ub))

        expected_ub_slope = 1 / (1 - (-2))
        expected_ub_intercept = -(-2) * expected_ub_slope

        assert layer.input_bounds  # mypy
        lb_slope = layer._get_lower_approximation_slopes(
            make_backsubstitution_config(), layer.input_bounds
        )
        ub_slope = layer._get_upper_approximation_slopes(
            make_backsubstitution_config(), layer.input_bounds
        )
        lb_intercept, ub_intercept = layer._get_approximation_intercepts(
            layer.input_bounds
        )

        assert (lb_slope == 0).all()
        assert (ub_slope == expected_ub_slope).all()
        assert (lb_intercept == 0).all()
        assert (ub_intercept == expected_ub_intercept).all()

    @set_torch_precision(torch.float32)
    def test_mn_bab_backsubstitution_coef(self) -> None:
        layer = ReLU((2,))
        layer_lb = torch.full(size=(1, 2), fill_value=-2.0)
        layer_ub = torch.full(size=(1, 2), fill_value=1.0)
        layer.update_input_bounds((layer_lb, layer_ub))

        lb = AffineForm(torch.eye(2).unsqueeze(0))
        ub = AffineForm(torch.tensor([[1.0, 0.0], [0.0, -1.0]]).unsqueeze(0))
        initial_shape = MN_BaB_Shape(
            query_id=query_tag(layer),
            query_prev_layer=None,
            queries_to_compute=None,
            lb=lb,
            ub=ub,
            unstable_queries=None,
            subproblem_state=None,
        )
        resulting_shape = layer.backsubstitute(
            make_backsubstitution_config(), initial_shape
        )

        expected_lb_slope = 0
        expected_ub_slope = 1 / (1 - (-2))

        lb_coef_as_expected = resulting_shape.lb.coef == torch.zeros(
            2, 2
        ).fill_diagonal_(expected_lb_slope)
        assert isinstance(lb_coef_as_expected, Tensor)
        assert lb_coef_as_expected.all()
        assert resulting_shape.ub is not None
        assert isinstance(resulting_shape.ub.coef, Tensor)
        assert resulting_shape.ub.coef[0, 0, 0] == expected_ub_slope
        assert resulting_shape.ub.coef[0, 1, 1] == expected_lb_slope

    @set_torch_precision(torch.float32)
    def test_mn_bab_backsubstitution_bias(self) -> None:
        layer_lb = torch.full(size=(1, 2), fill_value=-2.0)
        layer_ub = torch.full(size=(1, 2), fill_value=1.0)
        layer = ReLU((2,))

        lb = AffineForm(torch.eye(2).unsqueeze(0))
        ub = AffineForm(torch.tensor([[1.0, 0.0], [0.0, -1.0]]).unsqueeze(0))
        initial_shape = MN_BaB_Shape(
            query_id=query_tag(layer),
            query_prev_layer=None,
            queries_to_compute=None,
            lb=lb,
            ub=ub,
            unstable_queries=None,
            subproblem_state=None,
        )
        layer.update_input_bounds((layer_lb, layer_ub))

        resulting_shape = layer.backsubstitute(
            make_backsubstitution_config(), initial_shape
        )

        expected_lb_intercept = 0
        expected_ub_slope = 1 / (1 - (-2))
        expected_ub_intercept = -(-2) * expected_ub_slope

        assert (resulting_shape.lb.bias == expected_lb_intercept).all()
        assert resulting_shape.ub is not None
        assert resulting_shape.ub.bias[0, 0] == expected_ub_intercept
        assert resulting_shape.ub.bias[0, 1] == expected_lb_intercept

    @set_torch_precision(torch.float32)
    def test_propagate_abs(self) -> None:
        in_channels = 1
        input_dim = (in_channels, 5, 3)
        batch_size = 2

        layer = ReLU(input_dim)

        x = torch.rand((batch_size, *input_dim))
        x_out = layer(x)

        in_zono = HybridZonotope.construct_from_noise(
            x, eps=0.01, domain="zono", data_range=(-torch.inf, torch.inf)
        )
        out_zono = layer.propagate_abstract_element(in_zono)
        assert out_zono.shape == x_out.shape
        assert out_zono.may_contain_point(x_out)

        in_dpf = DeepPoly_f.construct_from_noise(
            x, eps=0.01, domain="DPF", data_range=(-torch.inf, torch.inf)
        )
        out_dpf = layer.propagate_abstract_element(in_dpf)
        assert out_dpf.shape == x_out.shape
        assert out_dpf.may_contain_point(x_out)

    @set_torch_precision(torch.float32)
    def test_propagate_interval_toy_example(self) -> None:
        layer = ReLU((2,))

        input_lb = torch.tensor([-2, -2])
        input_ub = torch.tensor([2, 2])

        expected_output_lb = torch.tensor([0, 0])
        expected_output_ub = torch.tensor([2, 2])
        output_lb, output_ub = layer.propagate_interval((input_lb, input_ub))

        assert (output_lb == expected_output_lb).all()
        assert (output_ub == expected_output_ub).all()

    @set_torch_precision(torch.float32)
    def test_propagate_interval_forcing_lower_triangle_approximation(self) -> None:
        layer = ReLU((2,))

        input_lb = torch.tensor([-2, -2])
        input_ub = torch.tensor([2.1, 2.1])

        expected_output_lb = torch.tensor([0, 0])
        expected_output_ub = torch.tensor([2.1, 2.1])
        output_lb, output_ub = layer.propagate_interval((input_lb, input_ub))

        assert (output_lb == expected_output_lb).all()
        assert (output_ub == expected_output_ub).all()
