from typing import Callable, Tuple, Union

import torch
from torch import Tensor
from torch.distributions.beta import Beta

from src.abstract_domains.DP_f import DeepPoly_f
from src.abstract_domains.zonotope import HybridZonotope
from src.utilities.initialization import seed_everything
from tests.test_util import _pgd_whitebox


def check_bounds(
    lb: Tensor,
    ub: Tensor,
    input_lb: Tensor,
    input_ub: Tensor,
    x: Tensor,
    net: Callable[[Tensor], Tensor],
    use_beta: bool = True,
    use_adv: bool = True,
) -> None:
    seed = 42
    device = input_lb.device
    input_shape = x.shape

    if use_beta:
        seed_everything(seed)
        m = Beta(concentration0=0.5, concentration1=0.5)
        eps = (input_ub - input_lb) / 2
        out = net(x)
        lb, ub = lb.to(device), ub.to(device)
        for i in range(100):
            shape_check = (256, *input_shape[1:])
            check_x = input_lb + 2 * eps * m.sample(shape_check).to(device)
            out = net(check_x)
            assert (lb - 1e-7 <= out).all() and (out <= ub + 1e-7).all()

    if use_adv:
        bounds = (lb.to(device), ub.to(device))
        target = torch.argmax(net(x)).item()
        _pgd_whitebox(
            net,  # type: ignore # slight abuse of passing a single layer instead of network
            x,
            bounds,
            target,
            input_lb,
            input_ub,
            x.device,
            num_steps=200,
        )


def einsum_test(
    defining_str: str,
    a_shape: Tuple[int, ...],
    b_shape: Tuple[int, ...],
    eps: float,
    n: int = 1,
) -> None:
    def einsum_layer(x: Tensor) -> Tensor:
        return torch.einsum(defining_str, x, b)

    for _ in range(n):
        a_input = torch.rand(a_shape)

        b = torch.rand(b_shape)

        a_abs: Union[HybridZonotope, DeepPoly_f] = HybridZonotope.construct_from_bounds(
            a_input - eps, a_input + eps, domain="zono"
        )
        output_shape = a_abs.einsum(defining_str, b)
        lb, ub = output_shape.concretize()
        check_bounds(
            lb,
            ub,
            a_input - eps,
            a_input + eps,
            a_input,
            einsum_layer,
            use_beta=True,
            use_adv=True,
        )

        a_abs = HybridZonotope.construct_from_bounds(
            a_input - eps, a_input + eps, domain="box"
        )
        output_shape = a_abs.einsum(defining_str, b)
        lb, ub = output_shape.concretize()
        check_bounds(
            lb,
            ub,
            a_input - eps,
            a_input + eps,
            a_input,
            einsum_layer,
            use_beta=True,
            use_adv=True,
        )

        a_abs = DeepPoly_f.construct_from_bounds(
            a_input - eps, a_input + eps, domain="DPF"
        )
        output_shape = a_abs.einsum(defining_str, b)
        lb, ub = output_shape.concretize()
        check_bounds(
            lb,
            ub,
            a_input - eps,
            a_input + eps,
            a_input,
            einsum_layer,
            use_beta=True,
            use_adv=True,
        )


class TestEinsum:
    def test_einsums(self, eps: float = 0.001, n: int = 1) -> None:
        einsum_test("bs, bqs -> bq", (1, 5), (1, 3, 5), eps, n)
        einsum_test("bsq, bqs -> bq", (1, 5, 3), (1, 3, 5), eps, n)
        einsum_test("blj, blk -> bk", (1, 3, 5), (1, 3, 4), eps, n)


if __name__ == "__main__":
    T = TestEinsum()
    n = 1
    T.test_einsums(0.001, n)
    T.test_einsums(0.01, n)
