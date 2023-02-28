import time

import torch
from torch.distributions.beta import Beta

from src.abstract_domains.DP_f import DeepPoly_f
from src.abstract_domains.zonotope import HybridZonotope
from src.milp_network import MILPNetwork
from src.state.tags import layer_tag
from src.utilities.initialization import seed_everything
from tests.test_util import (
    get_deep_poly_bounds,
    pad_toy_max_pool_net,
    toy_avg_pool_net,
    toy_max_avg_pool_net,
    toy_max_pool_net,
)


class TestPooling:
    """
    We test our Avg and Max-Pooling layers
    """

    def test_avg_pooling_sound(self) -> None:

        as_net, shape = toy_avg_pool_net()
        eps = 2 / 255
        m = Beta(concentration0=0.5, concentration1=0.5)
        for i in range(20):
            x = torch.rand(shape)
            lb = x - eps
            ub = x + eps
            (dp_lb, dp_ub) = get_deep_poly_bounds(as_net, lb, ub)

            in_zono = HybridZonotope.construct_from_bounds(lb, ub, domain="zono")
            out_zono = as_net.propagate_abstract_element(in_zono)

            in_dpf = DeepPoly_f.construct_from_bounds(lb, ub, domain="DPF")
            out_dpf = as_net.propagate_abstract_element(in_dpf)

            for _ in range(10):
                shape_check = (256, *shape[1:])
                check_x = lb + 2 * eps * m.sample(shape_check)
                out = as_net(check_x)
                assert (dp_lb <= out).all() and (out <= dp_ub).all()
                assert out_zono.may_contain_point(out)
                assert out_dpf.may_contain_point(out)

    def test_avg_pooling_milp(self) -> None:

        as_net = toy_avg_pool_net()[0]
        shape = (1, 1, 6, 6)
        eps = 0.25
        x = torch.ones(shape) * 0.5
        input_lb = x - eps
        input_ub = x + eps

        milp_model = MILPNetwork.build_model_from_abstract_net(
            x, input_lb, input_ub, as_net
        )
        for i, layer in enumerate(milp_model.net.layers):
            lbs, ubs = milp_model.get_network_bounds_at_layer_multi(
                layer_tag(layer),
                timeout_per_instance=20,
                timeout_total=400,
                timeout=time.time() + 400,
            )
            print(f"Layer {i} - LBS: {lbs} UBS: {ubs}")
            assert (lbs >= milp_model.net.layers[i].output_bounds[0].flatten()).all()
            assert (ubs <= milp_model.net.layers[i].output_bounds[1].flatten()).all()

    def test_max_pooling_milp(self) -> None:
        seed_everything(42)
        as_net = toy_max_pool_net()[0]
        shape = (1, 1, 6, 6)
        eps = 0.25
        x = torch.rand(shape) * 0.5

        input_lb = x - eps
        input_ub = x + eps

        milp_model = MILPNetwork.build_model_from_abstract_net(
            x, input_lb, input_ub, as_net
        )
        for i, layer in enumerate(milp_model.net.layers):
            lbs, ubs = milp_model.get_network_bounds_at_layer_multi(
                layer_tag(layer),
                timeout_per_instance=20,
                timeout_total=400,
                timeout=time.time() + 400,
            )
            print(f"Layer {i} - LBS: {lbs} UBS: {ubs}")
            assert (lbs >= milp_model.net.layers[i].output_bounds[0].flatten()).all()
            assert (ubs <= milp_model.net.layers[i].output_bounds[1].flatten()).all()

    def test_max_pooling_dp(self) -> None:

        seed_everything(42)
        as_net, shape = pad_toy_max_pool_net()
        eps = 0.25
        x = torch.ones(1, *shape) * 0.5

        input_lb = x - eps
        input_ub = x + eps

        milp_model = MILPNetwork.build_model_from_abstract_net(
            x, input_lb, input_ub, as_net
        )
        milp_lb, milp_ub = milp_model.get_network_output_bounds()
        (dp_lb, dp_ub) = get_deep_poly_bounds(as_net, input_lb, input_ub)

        assert (dp_lb <= milp_lb).all()
        assert (dp_ub >= milp_ub).all()
        m = Beta(concentration0=0.5, concentration1=0.5)
        eps = 8 / 255
        for i in range(100):
            x = torch.rand(1, *shape)
            input_lb = x - eps
            input_ub = x + eps
            as_net = pad_toy_max_pool_net()[0]
            (dp_lb, dp_ub) = get_deep_poly_bounds(as_net, input_lb, input_ub)

            # in_zono = HybridZonotope.construct_from_bounds(
            #     input_lb, input_ub, domain="zono"
            # )
            # out_zono = as_net.propagate_abstract_element(in_zono)

            # in_dpf = DeepPoly_f.construct_from_bounds(input_lb, input_ub, domain="DPF")
            # out_dpf = as_net.propagate_abstract_element(in_dpf)

            milp_model = MILPNetwork.build_model_from_abstract_net(
                x, input_lb, input_ub, as_net
            )
            out = as_net(x)
            assert (out >= dp_lb).all()
            assert (out <= dp_ub).all()
            for j in range(10):
                shape_check = (256, *shape)
                check_x = input_lb + 2 * eps * m.sample(shape_check)
                out = as_net(check_x)
                assert (dp_lb - 1e-7 <= out).all() and (out <= dp_ub + 1e-7).all()
                # assert out_zono.may_contain_point(out)
                # assert out_dpf.may_contain_point(out)

    def test_max_avg_pooling_milp(self) -> None:

        seed_everything(41)

        shape = (1, 1, 16, 16)
        eps = 2 / 255
        m = Beta(concentration0=0.5, concentration1=0.5)
        for i in range(5):
            x = torch.rand(shape)
            input_lb = x - eps
            input_ub = x + eps
            as_net = toy_max_avg_pool_net()
            milp_model = MILPNetwork.build_model_from_abstract_net(
                x, input_lb, input_ub, as_net
            )
            f_lbs, f_ubs = milp_model.get_network_output_bounds()
            for _ in range(20):
                shape_check = (256, *shape[1:])
                check_x = input_lb + 2 * eps * m.sample(shape_check)
                out = as_net(check_x)
                assert (f_lbs <= out).all() and (out <= f_ubs).all()


if __name__ == "__main__":
    t = TestPooling()
    t.test_max_pooling_dp()
    # t.test_avg_pooling_milp()
    # t.test_max_pooling_milp()
    # t.test_max_avg_pooling_milp()
