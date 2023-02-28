import gzip

import numpy as np

# import onnx
import onnxruntime as ort  # type: ignore[import]
import torch

from src.utilities.initialization import seed_everything
from src.utilities.loading.network import load_onnx_model


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

    def test_onnx_read(self) -> None:
        torch.set_default_dtype(torch.float32)
        net = load_onnx_model("benchmarks_vnn21/cifar10_resnet/onnx/resnet_2b.onnx")[0]

        x = torch.rand((1, 3, 32, 32))

        ort_session = ort.InferenceSession(
            "benchmarks_vnn21/cifar10_resnet/onnx/resnet_2b.onnx"
        )

        outputs = ort_session.run(
            None,
            {"input.1": np.array(x).astype(np.float32)},
        )
        print(outputs[0])

        net.eval()
        out = net(x)
        assert torch.isclose(torch.Tensor(outputs[0]), out, atol=1e-5).all()

    def test_onnx_read_gz(self) -> None:

        path = "vnn-comp-2022-sup/benchmarks/sri_resnet_a/onnx/resnet_3b2_bn_mixup_adv_4.0_bs128_lr-1.onnx"
        net, in_shape, in_name = load_onnx_model(path)
        x = torch.rand((1, *in_shape))

        if path.endswith(".gz"):
            ort_session = ort.InferenceSession(gzip.open(path).read())
        else:
            ort_session = ort.InferenceSession(path)

        outputs = ort_session.run(
            None,
            {in_name: np.array(x).astype(np.float32)},
        )
        print(outputs[0])

        net.eval()
        out = net(x)
        assert torch.isclose(torch.Tensor(outputs[0]), out, atol=1e-5).all()

    def test_onnx_nn4sys_2022(self) -> None:

        seed_everything(42)

        mscn_128d_path = "vnn-comp-2022-sup/benchmarks/nn4sys/onnx/mscn_128d.onnx"
        mscn_128d_dual_path = (
            "vnn-comp-2022-sup/benchmarks/nn4sys/onnx/mscn_128d_dual.onnx"
        )
        net, _, _ = load_onnx_model(mscn_128d_path)
        net.eval()
        if mscn_128d_path.endswith(".gz"):
            ort_session = ort.InferenceSession(gzip.open(mscn_128d_path).read())
        else:
            ort_session = ort.InferenceSession(mscn_128d_path)
        for i in range(20):
            x = torch.rand((1, 11, 14))
            outputs = ort_session.run(
                None,
                {"modelInput": np.array(x).astype(np.float32)},
            )
            out = net(x)
            assert torch.isclose(torch.Tensor(outputs[0]), out, atol=1e-5).all()

        net, _, _ = load_onnx_model(mscn_128d_dual_path)
        net.eval()
        if mscn_128d_dual_path.endswith(".gz"):
            ort_session = ort.InferenceSession(gzip.open(mscn_128d_dual_path).read())
        else:
            ort_session = ort.InferenceSession(mscn_128d_dual_path)

        # skl_model = skl_load_onnx_model("vnn-comp-2022-sup/benchmark_vnn22/nn4sys2022/model/mscn_128d_dual.onnx")
        # interm_model = select_model_inputs_outputs(skl_model, "138")
        # save_onnx_model(interm_model, "interm.onnx")
        # interm_session = ort.InferenceSession(
        #     "interm.onnx"
        # )
        # res = interm_session.run(None,
        #         {"modelInput": np.array(x).astype(np.float32)},
        # )
        # print(torch.tensor(res[0]))
        # net(x)

        for i in range(20):
            x = torch.rand((1, 22, 14))
            outputs = ort_session.run(
                None,
                {"modelInput": np.array(x).astype(np.float32)},
            )

            out = net(x)
            # print(f"onnx: {torch.Tensor(outputs[0]).item():.6f} o2p: {out.item():.6f}")
            assert torch.isclose(torch.Tensor(outputs[0]), out, atol=1e-5).all()

    def test_onnx_unet_2022(self) -> None:
        # Note this only works with skl2onnx installed
        pass
        # seed_everything(42)

        # path = "vnn-comp-2022-sup/benchmarks/carvana_unet_2022/onnx/unet_upsample_small.onnx.gz"
        # net, _, _ = load_onnx_model(path)
        # net.eval()

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

    def test_onnx_nn4sys(self) -> None:
        load_onnx_model("benchmarks_vnn21/nn4sys/nets/lognormal_100.onnx")
        load_onnx_model("benchmarks_vnn21/nn4sys/nets/lognormal_1000.onnx")
        # q3 = load_onnx_model("benchmarks_vnn21/nn4sys/nets/normal_100.onnx.gz")[0]
        # q4 = load_onnx_model("benchmarks_vnn21/nn4sys/nets/normal_1000.onnx.gz")[0]
        # q5 = load_onnx_model("benchmarks_vnn21/nn4sys/nets/piecewise_100.onnx.gz")[0]
        # q6 = load_onnx_model("benchmarks_vnn21/nn4sys/nets/piecewise_1000.onnx.gz")[0]
        print("Done")


if __name__ == "__main__":
    t = TestONNX()
    t.test_onnx_read()
    # t.test_onnx_unet_2022()
    # t.test_onnx_nn4sys_2022()
    # t.test_onnx_nn4sys()
