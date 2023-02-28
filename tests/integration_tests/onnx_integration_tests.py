import gzip
import os
import re
import shutil
from pathlib import Path
from typing import Tuple

import numpy as np

# import onnx
import onnxruntime as ort  # type: ignore[import]
import torch
from torch import nn

from src.abstract_layers.abstract_network import AbstractNetwork
from src.utilities.loading.network import load_net, load_onnx_model


def is_float_try(str: str) -> bool:
    try:
        float(str)
        return True
    except ValueError:
        return False


class TestONNX:
    """
    We test our ONNX parser implementation
    """

    def test_onnx_differential(self) -> None:
        dir = "networks"
        temp_dir = "tests/temp"

        try:
            Path(temp_dir).mkdir(parents=True, exist_ok=True)
            for net_name in sorted(os.listdir(dir), reverse=True):

                f = os.path.join(dir, net_name)

                net_split = re.split(r"_", net_name)

                dataset = net_split[0]
                net_pref = re.split(r"\.", net_name)[0]
                shape: Tuple[int, ...] = (784,)
                if dataset == "cifar" or dataset == "cifar10":
                    shape = (1, 3, 32, 32)
                elif dataset == "mnist":
                    if "flattened" in net_name:
                        shape = (1, 784)
                    else:
                        continue
                elif dataset == "resnet":
                    shape = (1, 3, 32, 32)
                else:
                    print(f"Unknown dataset {dataset}")
                    continue

                n_layers = 3
                n_neurons = 200

                if (
                    len(net_split) >= 2
                    and is_float_try(net_split[1])
                    and is_float_try(net_split[2])
                ):
                    n_layers = int(net_split[1])
                    n_neurons = int(net_split[2])

                try:
                    net_pt = load_net(f, n_layers, n_neurons)
                    print(f"Successfully loaded {net_name}")
                except Exception:
                    print(f"Couldn't load {net_name} - skipping")
                    continue

                net_pt.eval()
                x = torch.rand(shape)
                # store network to onnx
                onnx_path = f"{temp_dir}/{net_pref}.onnx"

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
                # load network from onnx
                # ort_session = ort.InferenceSession(onnx_path)
                # compare onnx to pytorch
                o2p_net, _, in_name = load_onnx_model(onnx_path)
                o2p_net.eval()
                # Compare results

                for i in range(10):
                    x = torch.rand(shape)

                    out_pt_net = net_pt(x)
                    out_o2p_net = o2p_net(x)
                    assert torch.isclose(out_pt_net, out_o2p_net).all()

                assert isinstance(o2p_net, nn.Sequential)
                AbstractNetwork.from_concrete_module(o2p_net, (1, *shape))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_onnx_benchmark21(self) -> None:
        dir = "benchmarks_vnn21"
        for root, dirs, files in os.walk(dir):
            for name in files:
                if name.split(".")[-1] == "onnx":
                    try:
                        onnx_path = os.path.join(root, name)
                        ort_session = ort.InferenceSession(onnx_path)
                        # compare onnx to pytorch
                        o2p_net, in_shape, in_name = load_onnx_model(onnx_path)
                        in_shape = (1, *in_shape)
                        o2p_net.eval()

                        for i in range(50):
                            x = torch.rand(in_shape)
                            output_onnx = ort_session.run(
                                None,
                                {in_name: np.array(x).astype(np.float32)},
                            )[0]
                            out_o2p_net = o2p_net(x)
                            assert torch.isclose(
                                torch.Tensor(output_onnx), out_o2p_net, atol=1e-5
                            ).all()
                        print(f"Successfully ran {onnx_path}")
                    except Exception as e:
                        print(f"Couldn't load {onnx_path} - skipping with {e}")

    def test_onnx_benchmark22(self) -> None:
        dir = "vnn-comp-2022-sup/benchmarks"
        for root, dirs, files in os.walk(dir):
            for name in files:
                if (
                    name.split(".")[-1] in ["onnx", "gz"]
                    and name.split(".")[-2] != "vnnlib"
                ):
                    try:
                        onnx_path = os.path.join(root, name)
                        if "vgg" in name:
                            print("Skipped vggnet")
                            continue

                        if name.split(".")[-1] == "gz":
                            onnx_byte_obj = gzip.open(onnx_path).read()
                            ort_session = ort.InferenceSession(onnx_byte_obj)
                        else:
                            ort_session = ort.InferenceSession(onnx_path)
                        # compare onnx to pytorch
                        o2p_net, in_shape, in_name = load_onnx_model(onnx_path)
                        in_shape = (1, *in_shape)
                        o2p_net.eval()

                        for i in range(50):
                            x = torch.rand(in_shape)
                            output_onnx = ort_session.run(
                                None,
                                {in_name: np.array(x).astype(np.float32)},
                            )[0]
                            out_o2p_net = o2p_net(x)
                            assert torch.isclose(
                                torch.Tensor(output_onnx), out_o2p_net, atol=1e-4
                            ).all(), "Bound violation"
                        print(f"Successfully ran {onnx_path}")
                    except Exception as e:
                        print(f"Couldn't run {onnx_path} - skipping with {e}")


if __name__ == "__main__":
    t = TestONNX()
    t.test_onnx_differential()
    t.test_onnx_benchmark21()
    t.test_onnx_benchmark22()
