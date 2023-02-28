import torch
import torch.nn as nn
from torch import Tensor

from src.abstract_layers.abstract_conv2d import Conv2d
from src.abstract_layers.abstract_max_pool2d import MaxPool2d
from src.abstract_layers.abstract_network import AbstractNetwork
from src.abstract_layers.abstract_relu import ReLU
from src.mn_bab_shape import AffineForm, MN_BaB_Shape
from src.state.tags import query_tag
from src.utilities.config import make_backsubstitution_config
from src.utilities.dependence_sets import DependenceSets
from src.utilities.initialization import seed_everything
from tests.test_util import toy_max_pool_mixed_net


class TestDependenceSets:
    def test_unfold_to_shapes(self) -> None:
        B, C, H, W = 10, 3, 4, 4
        c, h, w, d = 15, 13, 13, 7

        x = torch.rand((B, c, h, w))
        xs = [x, x.unsqueeze(1)]

        sets = torch.rand((B, C * H * W, c, d, d))
        idxs = torch.arange(H * W).repeat(C)
        coef = DependenceSets(
            sets=sets, spatial_idxs=idxs, input_dim=(C, H, W), cstride=3, cpadding=2
        )
        for x in xs:
            x_unfolded = DependenceSets.unfold_to(x, coef)
            assert list(x_unfolded.shape) == [B, C * H * W, c, d, d]

    def test_fold_to_shapes(self) -> None:
        B, C, H, W = 300, 5, 2, 2
        c, h, w, d = 70, 5, 5, 4
        old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)

        devices = ["cpu", "cuda"]

        for device in devices:
            x = torch.rand((B, c, h, w), device=device)

            sets = torch.rand((B, C * H * W, c, d, d), device=device)
            idxs = torch.arange(H * W, device=device).repeat(C)
            coef = DependenceSets(
                sets=sets, spatial_idxs=idxs, input_dim=(C, H, W), cstride=3, cpadding=2
            )
            x_unfolded = DependenceSets.unfold_to(x, coef)
            out_unfold = x_unfolded * coef.sets
            coef_tensor = coef.to_tensor((c, h, w))
            out_tensor = x.unsqueeze(1) * coef_tensor
            assert torch.isclose(
                out_unfold.flatten(2).sum(-1),
                out_tensor.flatten(2).sum(-1),
                atol=1e-12,
                rtol=1e-12,
            ).all(), f"failed for device {device}"
        torch.set_default_dtype(old_dtype)

    def test_concretize_shapes(self) -> None:
        B, C, H, W = 10, 3, 4, 4
        c, h, w, d = 15, 13, 13, 7

        input_bounds = torch.rand((c, h, w)).expand((B, c, h, w))

        sets = torch.rand((B, C * H * W, c, d, d))
        idxs = torch.arange(H * W).repeat(C)
        coef = DependenceSets(
            sets=sets, spatial_idxs=idxs, input_dim=(C, H, W), cstride=3, cpadding=2
        )
        bias = torch.rand((B, C * H * W))
        affine = AffineForm(coef, bias)
        output_lb, output_ub = MN_BaB_Shape(
            query_id=query_tag(ReLU((1,))),
            query_prev_layer=None,
            queries_to_compute=None,
            lb=affine,
            ub=affine,
            unstable_queries=None,
            subproblem_state=None,
        ).concretize(
            input_bounds,
            input_bounds,
        )
        assert list(output_lb.shape) == [B, C * H * W]
        assert output_ub is not None
        assert list(output_ub.shape) == [B, C * H * W]

    def test_conv2d_shapes(self) -> None:
        B, C, H, W = 10, 3, 4, 4
        c_pre, h_pre, w_pre = 8, 26, 26
        ksz, stride, padding = 4, 3, 2
        layer = nn.Conv2d(c_pre, 2 * c_pre, ksz, stride, padding)
        abstract_layer = Conv2d.from_concrete_module(layer, (c_pre, h_pre, w_pre))
        c, _, _ = abstract_layer.output_dim
        d = 3

        sets = torch.rand((B, C * H * W, c, d, d))
        idxs = torch.arange(H * W).repeat(C)
        coef = DependenceSets(
            sets=sets, spatial_idxs=idxs, input_dim=(C, H, W), cstride=3, cpadding=2
        )
        bias = torch.rand((B, C * H * W))
        affine = AffineForm(coef, bias)
        unstable_queries = torch.randint(0, 2, size=(C * H * W,), dtype=torch.bool)
        Q = unstable_queries.sum().item()
        affine = affine.filter_queries(unstable_queries)
        abstract_shape = MN_BaB_Shape(
            query_id=query_tag(ReLU((1,))),
            query_prev_layer=None,
            queries_to_compute=None,
            lb=affine,
            ub=affine,
            unstable_queries=unstable_queries,
            subproblem_state=None,
        )
        abstract_shape = abstract_layer.backsubstitute(
            make_backsubstitution_config(), abstract_shape
        )
        d_new = (d - 1) * stride + ksz
        assert isinstance(abstract_shape.lb.coef, DependenceSets)
        assert abstract_shape.ub is not None
        assert isinstance(abstract_shape.ub.coef, DependenceSets)
        for coef in [abstract_shape.lb.coef, abstract_shape.ub.coef]:
            assert all(
                [
                    type(coef) is DependenceSets,
                    list(coef.sets.shape) == [B, Q, c_pre, d_new, d_new],
                    coef.cstride == 3 * 3,
                    coef.cpadding == 3 * 2 + 2,
                ]
            )
        for bias in [abstract_shape.lb.bias, abstract_shape.ub.bias]:
            assert list(bias.shape) == [B, Q]

    def test_maxpool_shapes(self) -> None:
        device = torch.device("cuda")
        seed_everything(10)
        n = 2

        for _ in range(n):
            abstract_net, input_dim = toy_max_pool_mixed_net()
            abstract_net.layers = abstract_net.layers[:-1]
            abstract_net = abstract_net.to(device)
            abstract_net.output_dim = abstract_net.layers[-1].output_dim
            abstract_net.dependence_set_applicable = True
            c, H, W = abstract_net.output_dim
            d = 1
            B, C = 1, c

            e = 0.05
            input = torch.rand((B, *input_dim), device=device)
            tight_mask = torch.rand(input.shape, device=device) > 0.2
            input_lb = input - e
            input_ub = torch.where(tight_mask, input - e, input + e)

            sets = torch.rand((B, C * H * W, c, d, d), device=device)
            idxs = torch.arange(H * W, device=device).repeat(C)
            coef = DependenceSets(
                sets=sets, spatial_idxs=idxs, input_dim=(C, H, W), cstride=1, cpadding=0
            )
            bias = torch.rand((B, C * H * W), device=device)
            affine = AffineForm(coef, bias)
            abstract_shape = MN_BaB_Shape(
                query_id=query_tag(abstract_net),
                query_prev_layer=None,
                queries_to_compute=None,
                lb=affine.clone(),
                ub=affine.clone(),
                unstable_queries=None,
                subproblem_state=None,
            )

            abstract_shape_t = abstract_shape.clone_with_new_bounds(
                AffineForm(
                    abstract_shape.lb.coef.to_tensor(abstract_net.output_dim)
                    if isinstance(abstract_shape.lb.coef, DependenceSets)
                    else abstract_shape.lb.coef,
                    abstract_shape.lb.bias,
                ),
                None
                if abstract_shape.ub is None
                else AffineForm(
                    abstract_shape.ub.coef.to_tensor(abstract_net.output_dim)
                    if isinstance(abstract_shape.ub.coef, DependenceSets)
                    else abstract_shape.ub.coef,
                    abstract_shape.ub.bias,
                ),
            )

            abstract_shape_t = abstract_net.backsubstitute_mn_bab_shape(
                make_backsubstitution_config(
                    use_dependence_sets=False,
                    use_early_termination=False,
                ),
                input_lb,
                input_ub,
                query_coef=None,
                abstract_shape=abstract_shape_t,
                compute_upper_bound=True,
                reset_input_bounds=True,
                recompute_intermediate_bounds=True,
                optimize_intermediate_bounds=False,
            )
            output_lb_t, output_ub_t = abstract_shape_t.concretize(input_lb, input_ub)

            abstract_shape_ds = abstract_shape.clone_with_new_bounds(
                abstract_shape.lb.clone(),
                None if abstract_shape.ub is None else abstract_shape.ub.clone(),
            )
            abstract_shape_ds = abstract_net.backsubstitute_mn_bab_shape(
                make_backsubstitution_config(
                    use_dependence_sets=True,
                    use_early_termination=False,
                ),
                input_lb,
                input_ub,
                query_coef=None,
                abstract_shape=abstract_shape_ds,
                compute_upper_bound=True,
                reset_input_bounds=True,
                recompute_intermediate_bounds=True,
                optimize_intermediate_bounds=False,
            )
            output_lb_ds, output_ub_ds = abstract_shape_ds.concretize(
                input_lb, input_ub
            )

            assert torch.isclose(output_lb_ds, output_lb_t, atol=1e-10).all()
            assert output_ub_ds is not None
            assert output_ub_t is not None
            assert torch.isclose(output_ub_ds, output_ub_t, atol=1e-10).all()
        # assert isinstance(abstract_shape_t.lb.coef, Tensor)
        # assert torch.isclose(
        #     abstract_shape_t.lb.coef,
        #     abstract_shape_ds.lb.coef
        #     if isinstance(abstract_shape_ds.lb.coef, Tensor)
        #     else abstract_shape_ds.lb.coef.to_tensor(
        #         abstract_shape_t.lb.coef.shape[-3:]
        #     ),
        #     atol=1e-10,
        # ).all()

    def test_single_maxpool_shapes(self) -> None:
        device = torch.device("cuda")
        old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        seed_everything(10)

        c_pre, h_pre, w_pre = 2, 3, 3
        layer = nn.MaxPool2d(padding=1, stride=2, kernel_size=2)
        abstract_net = MaxPool2d.from_concrete_module(layer, (c_pre, h_pre, w_pre)).to(
            device
        )
        c, H, W = abstract_net.output_dim
        d = 1
        B = 1
        C = c

        e = 0.05
        input = torch.rand((B, *(c_pre, h_pre, w_pre)), device=device)
        tight_mask = torch.rand(input.shape, device=device) > 1.0
        input_lb = input - e
        input_ub = torch.where(tight_mask, input - e, input + e)

        abstract_net.input_bounds = (input_lb, input_ub)

        sets = torch.rand((B, C * H * W, c, d, d), device=device)
        idxs = torch.arange(H * W, device=device).repeat(C)
        coef = DependenceSets(
            sets=sets, spatial_idxs=idxs, input_dim=(C, H, W), cstride=1, cpadding=0
        )
        bias = torch.rand((B, C * H * W), device=device)
        affine = AffineForm(coef, bias)
        unstable_queries = torch.ones(
            size=(C * H * W,), dtype=torch.bool, device=device
        )
        abstract_shape = MN_BaB_Shape(
            query_id=query_tag(ReLU((1,))),
            query_prev_layer=None,
            queries_to_compute=None,
            lb=affine.clone(),
            ub=affine.clone(),
            unstable_queries=unstable_queries,
            subproblem_state=None,
        )
        config = make_backsubstitution_config()

        abstract_shape_t = abstract_shape.clone_with_new_bounds(
            AffineForm(
                abstract_shape.lb.coef.to_tensor(abstract_net.output_dim)
                if isinstance(abstract_shape.lb.coef, DependenceSets)
                else abstract_shape.lb.coef,
                abstract_shape.lb.bias,
            ),
            None
            if abstract_shape.ub is None
            else AffineForm(
                abstract_shape.ub.coef.to_tensor(abstract_net.output_dim)
                if isinstance(abstract_shape.ub.coef, DependenceSets)
                else abstract_shape.ub.coef,
                abstract_shape.ub.bias,
            ),
        )
        abstract_shape_t = abstract_net.backsubstitute(config, abstract_shape_t)

        abstract_shape_ds = abstract_shape.clone_with_new_bounds(
            abstract_shape.lb.clone(),
            None if abstract_shape.ub is None else abstract_shape.ub.clone(),
        )
        abstract_shape_ds = abstract_net.backsubstitute(config, abstract_shape_ds)

        assert torch.isclose(
            abstract_shape_t.lb.bias, abstract_shape_ds.lb.bias, atol=1e-10
        ).all()
        assert isinstance(abstract_shape_t.lb.coef, Tensor)
        assert torch.isclose(
            abstract_shape_t.lb.coef,
            abstract_shape_ds.lb.coef.to_tensor(abstract_shape_t.lb.coef.shape[-3:])
            if isinstance(abstract_shape_ds.lb.coef, DependenceSets)
            else abstract_shape_ds.lb.coef,
            atol=1e-10,
        ).all()
        torch.set_default_dtype(old_dtype)

    def test_multi_conv2d_shapes(self) -> None:
        device = torch.device("cuda")
        c_pre, h_pre, w_pre = 2, 6, 6
        layer_a = nn.Conv2d(c_pre, 2 * c_pre, 2, 1, 1)
        layer_b = nn.Conv2d(2 * c_pre, 3 * c_pre, 3, 2, 2)
        abstract_net = AbstractNetwork.from_concrete_module(
            nn.Sequential(layer_a, layer_b), (c_pre, h_pre, w_pre)
        ).to(device)
        c, _, _ = abstract_net.output_dim
        d = 1
        B, C, H, W = 1, 1, 2, 2

        sets = torch.rand((B, C * H * W, c, d, d), device=device)
        idxs = torch.arange(H * W, device=device).repeat(C)
        coef = DependenceSets(
            sets=sets, spatial_idxs=idxs, input_dim=(C, H, W), cstride=2, cpadding=0
        )
        bias = torch.rand((B, C * H * W), device=device)
        affine = AffineForm(coef, bias)
        unstable_queries = torch.ones(
            size=(C * H * W,), dtype=torch.bool, device=device
        )
        abstract_shape = MN_BaB_Shape(
            query_id=query_tag(ReLU((1,))),
            query_prev_layer=None,
            queries_to_compute=None,
            lb=affine.clone(),
            ub=affine.clone(),
            unstable_queries=unstable_queries,
            subproblem_state=None,
        )
        config = make_backsubstitution_config()

        assert isinstance(abstract_shape.lb.coef, DependenceSets)
        assert abstract_shape.ub is not None and isinstance(
            abstract_shape.ub.coef, DependenceSets
        )

        abstract_shape_t = abstract_shape.clone_with_new_bounds(
            AffineForm(
                abstract_shape.lb.coef.to_tensor(abstract_net.output_dim),
                abstract_shape.lb.bias,
            ),
            None
            if abstract_shape.ub is None
            else AffineForm(
                abstract_shape.ub.coef.to_tensor(abstract_net.output_dim),
                abstract_shape.ub.bias,
            ),
        )
        abstract_shape_t = abstract_net.layers[0].backsubstitute(
            config, abstract_net.layers[1].backsubstitute(config, abstract_shape_t)
        )

        abstract_shape_ds = abstract_shape.clone_with_new_bounds(
            abstract_shape.lb.clone(),
            None if abstract_shape.ub is None else abstract_shape.ub.clone(),
        )
        abstract_shape_ds = abstract_net.layers[0].backsubstitute(
            config, abstract_net.layers[1].backsubstitute(config, abstract_shape_ds)
        )

        assert torch.isclose(
            abstract_shape_t.lb.bias, abstract_shape_ds.lb.bias, atol=1e-10
        ).all()
        assert isinstance(abstract_shape_t.lb.coef, Tensor)
        assert isinstance(abstract_shape_ds.lb.coef, DependenceSets)

        assert torch.isclose(
            abstract_shape_t.lb.coef,
            abstract_shape_ds.lb.coef.to_tensor(abstract_shape_t.lb.coef.shape[-3:]),
            atol=1e-10,
        ).all()

    def test_relu_shapes(self) -> None:
        B, C, H, W = 10, 3, 4, 4
        c, h, w, d = 15, 13, 13, 7
        stride, padding = 3, 2
        abstract_layer = ReLU((c, h, w))
        input_bounds = torch.rand((B, c, h, w))
        abstract_layer.update_input_bounds((input_bounds, input_bounds))

        sets = torch.rand((B, C * H * W, c, d, d))
        idxs = torch.arange(H * W).repeat(C)
        coef = DependenceSets(
            sets=sets,
            spatial_idxs=idxs,
            input_dim=(C, H, W),
            cstride=stride,
            cpadding=padding,
        )
        bias = torch.rand((B, C * H * W))
        affine = AffineForm(coef, bias)
        unstable_queries = torch.randint(0, 2, size=(C * H * W,), dtype=torch.bool)
        Q = unstable_queries.sum()
        affine = affine.filter_queries(unstable_queries)
        abstract_shape = MN_BaB_Shape(
            query_id=query_tag(ReLU((1,))),
            query_prev_layer=None,
            queries_to_compute=None,
            lb=affine,
            ub=affine,
            unstable_queries=unstable_queries,
            subproblem_state=None,
        )
        abstract_shape = abstract_layer.backsubstitute(
            make_backsubstitution_config(), abstract_shape
        )
        assert isinstance(abstract_shape.lb.coef, DependenceSets)
        assert abstract_shape.ub is not None
        assert isinstance(abstract_shape.ub.coef, DependenceSets)
        for coef in [abstract_shape.lb.coef, abstract_shape.ub.coef]:
            assert all(
                [
                    type(coef) is DependenceSets,
                    list(coef.sets.shape) == [B, Q, c, d, d],
                    coef.cstride == stride,
                    coef.cpadding == padding,
                ]
            )
        for bias in [abstract_shape.lb.bias, abstract_shape.ub.bias]:
            assert list(bias.shape) == [B, Q]


if __name__ == "__main__":
    T = TestDependenceSets()
    T.test_single_maxpool_shapes()
    T.test_maxpool_shapes()
    T.test_multi_conv2d_shapes()
    T.test_fold_to_shapes()
    T.test_concretize_shapes()
    T.test_unfold_to_shapes()
    T.test_conv2d_shapes()
