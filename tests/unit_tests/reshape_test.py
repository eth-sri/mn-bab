import shutil
from pathlib import Path

import torch
from torch import nn
from torch.distributions import Beta

from src.abstract_domains.DP_f import DeepPoly_f
from src.abstract_domains.zonotope import HybridZonotope
from src.abstract_layers.abstract_network import AbstractNetwork
from src.utilities.loading.network import load_onnx_model
from tests.test_util import get_deep_poly_bounds, toy_reshape_net


class TestReshape:
    """
    We test our Reshape layer.
    """

    def test_reshape(self) -> None:

        net_pt = toy_reshape_net()
        shape = (256,)
        onnx_shape = (1, 256)
        eps = 2 / 255

        try:
            temp_dir = "tests/temp"
            net_pref = "reshape_test"
            onnx_path = f"{temp_dir}/{net_pref}.onnx"
            Path(temp_dir).mkdir(parents=True, exist_ok=True)

            x = torch.rand(onnx_shape)

            torch.onnx.export(
                net_pt,
                x,
                onnx_path,
                export_params=True,
                training=0,
                do_constant_folding=False,
                verbose=False,
                input_names=["input.1"],
                output_names=["output"],
            )

            o2p_net, _, in_name = load_onnx_model(onnx_path)
            o2p_net.eval()
            # Compare results

            for i in range(10):
                x = torch.rand(onnx_shape)

                out_pt_net = net_pt(x)
                out_o2p_net = o2p_net(x)
                assert torch.isclose(out_pt_net, out_o2p_net).all()

            # Get abstract net
            assert isinstance(o2p_net, nn.Sequential)
            abs_net = AbstractNetwork.from_concrete_module(o2p_net, shape)

            input = torch.rand(onnx_shape)
            input_lb = input - eps
            input_ub = input + eps
            (
                dp_lb,
                dp_ub,
            ) = get_deep_poly_bounds(abs_net, input_lb, input_ub)

            in_zono = HybridZonotope.construct_from_bounds(
                input_lb, input_ub, domain="zono"
            )
            out_zono = abs_net.propagate_abstract_element(in_zono)

            in_dpf = DeepPoly_f.construct_from_bounds(input_lb, input_ub, domain="DPF")
            out_dpf = abs_net.propagate_abstract_element(in_dpf)

            m = Beta(concentration0=0.5, concentration1=0.5)

            for _ in range(10):
                shape_check = (256, *shape[1:])
                check_x = input_lb + 2 * eps * m.sample(shape_check)
                out = abs_net(check_x)
                assert (dp_lb <= out).all() and (out <= dp_ub).all()
                assert out_zono.may_contain_point(out)
                assert out_dpf.may_contain_point(out)

            print("Done")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    T = TestReshape()
    T.test_reshape()
