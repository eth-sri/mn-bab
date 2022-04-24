import torch
import torch.nn as nn

from src.abstract_layers.abstract_conv2d import Conv2d
from src.abstract_layers.abstract_relu import ReLU
from src.mn_bab_shape import MN_BaB_Shape
from src.utilities.dependence_sets import DependenceSets


class TestDependenceSets:
    def test_unfold_to_shapes(self) -> None:
        B, C, HW = 10, 3, 16
        c, h, w, d = 15, 13, 13, 7

        x = torch.rand((B, c, h, w))
        xs = [x, x.unsqueeze(1)]

        sets = torch.rand((B, C * HW, c, d, d))
        idxs = torch.arange(HW).repeat(C)
        coef = DependenceSets(sets=sets, spatial_idxs=idxs, cstride=3, cpadding=2)
        for x in xs:
            x_unfolded = DependenceSets.unfold_to(x, coef)
            assert list(x_unfolded.shape) == [B, C * HW, c, d, d]

    def test_concretize_shapes(self) -> None:
        B, C, HW = 10, 3, 16
        c, h, w, d = 15, 13, 13, 7

        input_bounds = torch.rand((c, h, w))

        sets = torch.rand((B, C * HW, c, d, d))
        idxs = torch.arange(HW).repeat(C)
        coef = DependenceSets(sets=sets, spatial_idxs=idxs, cstride=3, cpadding=2)
        bias = torch.rand((B, C * HW))
        output_lb, output_ub = MN_BaB_Shape(coef, coef, bias, bias).concretize(
            input_bounds, input_bounds
        )
        assert list(output_lb.shape) == [B, C * HW]
        assert list(output_ub.shape) == [B, C * HW]

    def test_conv2d_shapes(self) -> None:
        B, C, HW = 10, 3, 16
        c, d = 15, 7
        ksz, stride, padding = 4, 3, 2
        c_pre, h_pre, w_pre = 8, 26, 26
        layer = nn.Conv2d(c_pre, c, ksz, stride, padding)
        abstract_layer = Conv2d.from_concrete_module(layer, (c_pre, h_pre, w_pre))

        sets = torch.rand((B, C * HW, c, d, d))
        idxs = torch.arange(HW).repeat(C)
        coef = DependenceSets(sets=sets, spatial_idxs=idxs, cstride=3, cpadding=2)
        bias = torch.rand((B, C * HW))
        abstract_shape = MN_BaB_Shape(coef, coef, bias, bias)
        unstable_queries = torch.randint(0, 2, size=(C * HW,), dtype=torch.bool)
        Q = unstable_queries.sum().item()
        abstract_shape.filter_out_stable_queries(unstable_queries)
        abstract_shape = abstract_layer.backsubstitute(abstract_shape)
        d_new = (d - 1) * stride + ksz
        for coef in [abstract_shape.lb_coef, abstract_shape.ub_coef]:
            assert all(
                [
                    type(coef) is DependenceSets,
                    list(coef.sets.shape) == [B, Q, c_pre, d_new, d_new],
                    coef.cstride == 3 * 3,
                    coef.cpadding == 3 * 2 + 2,
                ]
            )
        for bias in [abstract_shape.lb_bias, abstract_shape.ub_bias]:
            assert list(bias.shape) == [B, Q]

    def test_relu_shapes(self) -> None:
        B, C, HW = 10, 3, 16
        c, h, w, d = 15, 13, 13, 7
        stride, padding = 3, 2
        abstract_layer = ReLU((c, h, w))
        input_bounds = torch.rand((B, c, h, w))
        abstract_layer.update_input_bounds((input_bounds, input_bounds))

        sets = torch.rand((B, C * HW, c, d, d))
        idxs = torch.arange(HW).repeat(C)
        coef = DependenceSets(
            sets=sets, spatial_idxs=idxs, cstride=stride, cpadding=padding
        )
        bias = torch.rand((B, C * HW))
        abstract_shape = MN_BaB_Shape(coef, coef, bias, bias)
        unstable_queries = torch.randint(0, 2, size=(C * HW,), dtype=torch.bool)
        Q = unstable_queries.sum()
        abstract_shape.filter_out_stable_queries(unstable_queries)
        abstract_shape = abstract_layer.backsubstitute(abstract_shape)
        for coef in [abstract_shape.lb_coef, abstract_shape.ub_coef]:
            assert all(
                [
                    type(coef) is DependenceSets,
                    list(coef.sets.shape) == [B, Q, c, d, d],
                    coef.cstride == stride,
                    coef.cpadding == padding,
                ]
            )
        for bias in [abstract_shape.lb_bias, abstract_shape.ub_bias]:
            assert list(bias.shape) == [B, Q]
