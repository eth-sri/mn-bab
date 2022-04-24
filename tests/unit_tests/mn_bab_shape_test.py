import torch

from src.mn_bab_shape import MN_BaB_Shape


class TestMNBaBShape:
    def test_init_with_full_argument_list(self) -> None:
        lb_coef = torch.rand(1, 2, 2)
        ub_coef = torch.rand(1, 2, 2)
        lb_bias = torch.rand(2, 2)
        ub_bias = torch.rand(2, 2)
        MN_BaB_Shape(lb_coef, ub_coef, lb_bias, ub_bias)

    def test_init_with_partial_argument_list(self) -> None:
        lb_coef = torch.rand(1, 2, 2)
        ub_coef = torch.rand(1, 2, 2)
        MN_BaB_Shape(lb_coef, ub_coef)
