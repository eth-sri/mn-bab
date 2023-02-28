import torch
from torch import Tensor

from src.abstract_domains.DP_f import DeepPoly_f
from src.abstract_domains.zonotope import HybridZonotope
from src.abstract_layers.abstract_linear import Linear
from src.mn_bab_shape import AffineForm, MN_BaB_Shape
from src.state.tags import query_tag
from src.utilities.config import make_backsubstitution_config


class TestLinear:
    def test_backsubstitution_mn_bab(self) -> None:
        layer = Linear(10, 2, bias=True, input_dim=(10,))

        lb_coef = torch.eye(2).unsqueeze(0)
        lb = AffineForm(lb_coef)
        ub = AffineForm(2 * lb_coef)
        initial_shape = MN_BaB_Shape(
            query_id=query_tag(layer),
            query_prev_layer=None,
            queries_to_compute=None,
            lb=lb,
            ub=ub,
            unstable_queries=None,
            subproblem_state=None,
        )
        assert isinstance(initial_shape.lb.coef, Tensor)
        assert initial_shape.ub is not None
        assert isinstance(initial_shape.ub.coef, Tensor)

        layer = Linear(10, 2, bias=True, input_dim=(10,))

        expected_lb_coef = initial_shape.lb.coef.matmul(layer.weight)
        expected_lb_bias = initial_shape.lb.coef.matmul(layer.bias)

        expected_lb = AffineForm(expected_lb_coef, expected_lb_bias)

        expected_ub_coef = initial_shape.ub.coef.matmul(layer.weight)
        expected_ub_bias = initial_shape.ub.coef.matmul(layer.bias)

        expected_ub = AffineForm(expected_ub_coef, expected_ub_bias)

        expected_shape = MN_BaB_Shape(
            query_id=query_tag(layer),
            query_prev_layer=None,
            queries_to_compute=None,
            lb=expected_lb,
            ub=expected_ub,
            unstable_queries=None,
            subproblem_state=None,
        )
        assert isinstance(expected_shape.lb.coef, Tensor)
        assert expected_shape.ub is not None
        assert isinstance(expected_shape.ub.coef, Tensor)

        actual_shape = layer.backsubstitute(
            make_backsubstitution_config(), initial_shape
        )

        assert isinstance(actual_shape.lb.coef, Tensor)
        assert actual_shape.ub is not None
        assert isinstance(actual_shape.ub.coef, Tensor)

        assert expected_shape.lb.coef.equal(actual_shape.lb.coef)
        assert expected_shape.ub.coef.equal(actual_shape.ub.coef)
        assert expected_shape.lb.bias.equal(actual_shape.lb.bias)
        assert expected_shape.ub.bias.equal(actual_shape.ub.bias)

    def test_propagate_abs_linear(self) -> None:
        input_dim = (12,)
        batch_size = 2
        eps = 0.01

        layer = Linear(input_dim[0], 3, bias=True, input_dim=(12,))

        x = torch.rand((batch_size, *input_dim))
        x_out = layer(x)

        in_zono = HybridZonotope.construct_from_noise(x, eps=eps, domain="zono")
        out_zono = layer.propagate_abstract_element(in_zono)
        assert out_zono.shape == x_out.shape
        assert out_zono.may_contain_point(x_out)

        in_dpf = DeepPoly_f.construct_from_noise(x, eps=eps, domain="DPF")
        out_dpf = layer.propagate_abstract_element(in_dpf)
        assert out_dpf.shape == x_out.shape
        assert out_dpf.may_contain_point(x_out)

    def test_propagate_interval_identity_layer(self) -> None:
        layer = Linear(2, 2, bias=True, input_dim=(2,))
        layer.weight.data = torch.eye(2)
        layer.bias.data = torch.zeros(2)

        input_lb = torch.tensor([-1.0, -1.0])
        input_ub = torch.tensor([1.0, 1.0])

        output_lb, output_ub = layer.propagate_interval((input_lb, input_ub))

        assert (output_lb == input_lb).all()
        assert (output_ub == input_ub).all()

    def test_propagate_interval_toy_example_layer(self) -> None:
        layer = Linear(2, 2, bias=True, input_dim=(2,))
        layer.weight.data = torch.tensor([[1.0, 1.0], [1.0, -1.0]])
        layer.bias.data = torch.tensor([1, -1])

        input_lb = torch.tensor([-1.0, -1.0])
        input_ub = torch.tensor([1.0, 1.0])

        expected_output_lb = torch.tensor([-1, -3])
        expected_output_ub = torch.tensor([3, 1])
        output_lb, output_ub = layer.propagate_interval((input_lb, input_ub))

        assert (output_lb == expected_output_lb).all()
        assert (output_ub == expected_output_ub).all()
