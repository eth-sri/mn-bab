import gzip

import numpy as np
import onnxruntime as ort  # type: ignore[import]
import torch
from torch import nn

from src.abstract_layers.abstract_network import AbstractNetwork
from src.utilities.loading.network import load_onnx_model
from tests.test_util import get_deep_poly_bounds


def is_float_try(str: str) -> bool:
    try:
        float(str)
        return True
    except ValueError:
        return False


class TestVNN:
    """
    Integration test for the vnn22 benchmarks
    """

    def test_onnx_single_net(self) -> None:

        # Test ONNX Parse
        onnx_path = "vnn-comp-2022-sup/benchmarks/nn4sys/onnx/mscn_128d.onnx.gz"
        ort_session = ort.InferenceSession(gzip.open(onnx_path).read())
        o2p_net, in_shape, in_name = load_onnx_model(onnx_path)
        in_shape_batch = (1, *in_shape)
        o2p_net.eval()

        for i in range(50):
            x = torch.rand(in_shape_batch)
            output_onnx = ort_session.run(
                None,
                {in_name: np.array(x).astype(np.float32)},
            )[0]
            out_o2p_net = o2p_net(x)
            assert torch.isclose(
                torch.Tensor(output_onnx), out_o2p_net, atol=1e-5
            ).all()

        assert isinstance(o2p_net, nn.Sequential)

        # Run AS Parse
        abs_net = AbstractNetwork.from_concrete_module(o2p_net, in_shape)
        eps = 2 / 255

        # Run interval propagation
        in_shape_squeezed = tuple([1] + [i for i in in_shape_batch if i != 1])
        input = torch.rand(in_shape_squeezed)
        input_lb = input - eps
        input_ub = input + eps

        lb, ub = abs_net.set_layer_bounds_via_interval_propagation(
            input_lb, input_ub, use_existing_bounds=False, has_batch_dim=True
        )

        print(f"Found lower {lb} and upper {ub}")

        (
            output_lb_without_alpha,
            output_ub_without_alpha,
        ) = get_deep_poly_bounds(abs_net, input_lb, input_ub)

        print(f"Succesful run of {onnx_path} - LB {output_lb_without_alpha}")

    def test_onnx_dual_net(self) -> None:
        onnx_path = "vnn-comp-2022-sup/benchmarks/nn4sys/onnx/mscn_128d_dual.onnx.gz"
        ort_session = ort.InferenceSession(gzip.open(onnx_path).read())
        o2p_net, in_shape, in_name = load_onnx_model(onnx_path)
        in_shape_batch = (1, *in_shape)
        o2p_net.eval()

        for i in range(50):
            x = torch.rand(in_shape_batch)
            output_onnx = ort_session.run(
                None,
                {in_name: np.array(x).astype(np.float32)},
            )[0]
            out_o2p_net = o2p_net(x)
            assert torch.isclose(
                torch.Tensor(output_onnx), out_o2p_net, atol=1e-5
            ).all()

        assert isinstance(o2p_net, nn.Sequential)

        # Run AS Parse
        abs_net = AbstractNetwork.from_concrete_module(o2p_net, in_shape)
        eps = 2 / 255

        # Run interval propagation
        in_shape_squeezed = tuple([1] + [i for i in in_shape_batch if i != 1])
        input = torch.rand(in_shape_squeezed)
        input_lb = input - eps
        input_ub = input + eps

        lb, ub = abs_net.set_layer_bounds_via_interval_propagation(
            input_lb, input_ub, use_existing_bounds=False, has_batch_dim=True
        )

        print(f"Found lower {lb} and upper {ub}")

        (
            output_lb_without_alpha,
            output_ub_without_alpha,
        ) = get_deep_poly_bounds(abs_net, input_lb, input_ub)

        print(f"Succesful run of {onnx_path} - LB {output_lb_without_alpha}")

    def test_onnx_u_net(self) -> None:

        onnx_path = "vnn-comp-2022-sup/benchmarks/carvana_unet_2022/onnx/unet_upsample_small.onnx.gz"
        o2p_net, in_shape, in_name = load_onnx_model(onnx_path)
        in_shape_batch = (1, *in_shape)
        o2p_net.eval()

        # Run AS Parse
        abs_net = AbstractNetwork.from_concrete_module(o2p_net, in_shape)
        eps = 2 / 255

        # Run interval propagation
        input = torch.rand(in_shape_batch)
        input_lb = input - eps
        input_ub = input + eps

        lb, ub = abs_net.set_layer_bounds_via_interval_propagation(
            input_lb, input_ub, use_existing_bounds=False, has_batch_dim=True
        )

        print(f"Found lower {lb} and upper {ub}")

        (
            output_lb_without_alpha,
            output_ub_without_alpha,
        ) = get_deep_poly_bounds(abs_net, input_lb, input_ub)

        print(f"Succesful run of {onnx_path} - LB {output_lb_without_alpha}")

        # from skl2onnx.helpers.onnx_helper import load_onnx_model as skl_load_onnx_model
        # from skl2onnx.helpers.onnx_helper import (
        #     select_model_inputs_outputs,
        #     save_onnx_model,
        # )

        # skl_model = skl_load_onnx_model(gzip.open(path).read())
        # interm_model = select_model_inputs_outputs(skl_model, "out_mask")
        # save_onnx_model(interm_model, "interm.onnx")
        # interm_session = ort.InferenceSession("interm.onnx")

        # for _ in range(20):
        #     input = torch.rand(1, 4, 31, 47)

        #     out = net(input)

        #     res = interm_session.run(
        #         None,
        #         {"input": np.array(input).astype(np.float32)},
        #     )
        #     res = torch.tensor(res[0])
        #     assert torch.isclose(res, out, atol=1e-5).all()


if __name__ == "__main__":
    t = TestVNN()
    # t.test_onnx_u_net()
    t.test_onnx_single_net()
    t.test_onnx_dual_net()
