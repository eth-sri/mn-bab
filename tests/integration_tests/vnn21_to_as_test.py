import os

import numpy as np
import onnxruntime as ort  # type: ignore[import]
import pytest
import torch
from torch import nn
from torch.distributions.beta import Beta

from src.abstract_layers.abstract_network import AbstractNetwork
from src.utilities.initialization import seed_everything
from src.utilities.loading.network import load_onnx_model
from tests.test_util import get_deep_poly_bounds, get_deep_poly_lower_bounds


def is_float_try(str: str) -> bool:
    try:
        float(str)
        return True
    except ValueError:
        return False


class TestVNN:
    """
    Integration test for the vnn21 benchmarks
    """

    def test_onnx_specific_net(self) -> None:

        onnx_path = "benchmarks_vnn21/marabou-cifar10/nets/cifar10_small.onnx"
        ort_session = ort.InferenceSession(onnx_path)
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
        abs_net = AbstractNetwork.from_concrete_module(o2p_net, in_shape)
        eps = 2 / 255
        in_shape_squeezed = tuple([1] + [i for i in in_shape_batch if i != 1])

        input = torch.rand(in_shape_squeezed)
        input_lb = input - eps
        input_ub = input + eps
        (
            output_lb_without_alpha,
            output_ub_without_alpha,
        ) = get_deep_poly_bounds(abs_net, input_lb, input_ub)

        print(f"Succesful run of {onnx_path} - LB {output_lb_without_alpha}")

    # Creating a coverage file here requires too much RAM
    @pytest.mark.skip(reason="Creating a coverage file here requires too much RAM")
    def test_onnx_to_abstract_net_benchmark(self) -> None:

        seed_everything(42)
        dir = "benchmarks_vnn21"
        eps = 0.1
        m = Beta(concentration0=0.5, concentration1=0.5)

        for root, dirs, files in os.walk(dir):
            for name in files:
                if name.split(".")[-1] == "onnx":
                    if name in [
                        "cifar10_2_255_simplified.onnx",
                        "cifar10_2_255.onnx",
                        "cifar10_8_255.onnx",
                        "convBigRELU__PGD.onnx",
                        "Convnet_maxpool.onnx",
                    ]:
                        continue
                    try:

                        onnx_path = os.path.join(root, name)

                        # compare onnx to pytorch
                        o2p_net, in_shape, in_name = load_onnx_model(onnx_path)
                        o2p_net.eval()

                        assert isinstance(o2p_net, nn.Sequential)
                        abs_net = AbstractNetwork.from_concrete_module(
                            o2p_net, in_shape
                        )
                        in_shape_squeezed = tuple([1] + [i for i in in_shape if i != 1])
                        batch_in_shape = (1, *in_shape)
                        input = torch.rand(in_shape_squeezed)
                        input_lb = input - eps
                        input_ub = input + eps
                        (
                            output_lb_without_alpha,
                            output_ub_without_alpha,
                        ) = get_deep_poly_bounds(abs_net, input_lb, input_ub)
                        only_lb, _ = get_deep_poly_lower_bounds(
                            abs_net, input_lb, input_ub
                        )
                        assert (output_lb_without_alpha == only_lb).all()
                        shape_check = (256, *in_shape)
                        if (
                            len(input_lb.shape) < len(batch_in_shape)
                            and input_lb.shape == batch_in_shape[: len(input_lb.shape)]
                        ):
                            input_lb = input_lb.unsqueeze(-1)

                        for _ in range(2):
                            check_x = input_lb.broadcast_to(
                                batch_in_shape
                            ) + 2 * eps * m.sample(shape_check)
                            if len(check_x.shape) == 1:  # Special case for Nano input
                                check_x = check_x.unsqueeze(1)
                            out = o2p_net(check_x)
                            assert (output_lb_without_alpha <= out).all() and (
                                out <= output_ub_without_alpha
                            ).all()

                        print(f"Succesful run of {onnx_path}")
                    except Exception as e:
                        assert False, f"Couldn't run {onnx_path} - skipping with: {e}"


if __name__ == "__main__":
    t = TestVNN()
    t.test_onnx_specific_net()
    t.test_onnx_to_abstract_net_benchmark()
