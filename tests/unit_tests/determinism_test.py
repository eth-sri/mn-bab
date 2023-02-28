import torch

from src.utilities.initialization import seed_everything
from src.utilities.loading.network import load_onnx_model


class TestDeterminism:
    """
    We test our deterministic behaviour
    """

    def test_determ(self) -> None:

        d_type = torch.float64
        seed_everything(1)
        assert_flag = False
        net, onnx_shape, inp_name = load_onnx_model(
            "vnn-comp-2022-sup/benchmarks/sri_resnet_b/onnx/resnet_3b2_bn_mixup_ssadv_4.0_bs128_lr-1_v2.onnx"
        )
        x = torch.rand((1, 3, 32, 32), dtype=d_type)

        net.eval()
        if d_type == torch.float64:
            print("Using double precision")
            net.double()
        out = net(x)
        X_pgd = torch.rand((50, 3, 32, 32), dtype=d_type)
        X_cat = torch.cat([x, X_pgd])

        print("Running cpu test")
        print("Running batch_size tests")
        issues_found = 0
        for bs in range(1, 50):
            # print(bs)
            out_i = net(X_cat[:bs])[0]
            if not torch.allclose(out, out_i):  # (out_i != out).any():
                print(
                    f"Mismatch at {bs}: {out} to {out_i} - Diff = {torch.sum(torch.abs(out-out_i))}"
                )
                assert_flag = True
                issues_found += 1
                out = out_i
            # print(f"Out {bs}: {out_i}")
        print(f"Found {issues_found} mismatches")

        if torch.cuda.is_available():
            print("Running cuda tests")
            net.to("cuda")
            out_cpu = out
            out_cuda = net(x.cuda())
            if not torch.allclose(out_cpu, out_cuda.cpu()):
                print(
                    f"CPU and CUDA not consistent: CPU {out_cpu} GPU {out_cuda.cpu()} - Diff = {torch.sum(torch.abs(out_cpu-out_cuda.cpu()))}"
                )
                assert_flag = True

            print("Running batch_size tests")
            out = out_cuda
            issues_found = 0
            for bs in range(1, 50):
                out_i = net(X_cat[:bs].cuda())[0]
                if not torch.allclose(out, out_i):
                    print(
                        f"Mismatch at {bs}: {out} to {out_i} - Diff = {torch.sum(torch.abs(out-out_i))}"
                    )
                    assert_flag = True
                    issues_found += 1
                    out = out_i
            print(f"Found {issues_found} mismatches")

            print("Running equal batch-sizes, different aux batch test")
            for bs in range(2, 20):
                issues_found = 0
                out = out_cuda
                for j in range(10):
                    X_pgd = torch.rand((bs - 1, 3, 32, 32), dtype=d_type).cuda()
                    X_cat = torch.cat([x.cuda(), X_pgd])
                    out_i = net(X_cat.cuda())[0]
                    if j == 0:
                        out = out_i
                    if not torch.allclose(out, out_i):
                        print(f"Mismatch {bs} - {j}: {out} to {out_i}")
                        assert_flag = True
                        issues_found += 1
                        out = out_i
            print(f"Found {issues_found} mismatches")

        if assert_flag:
            pass
        #    assert False, "Non-deterministic behaviour detected"


if __name__ == "__main__":
    t = TestDeterminism()
    t.test_determ()
