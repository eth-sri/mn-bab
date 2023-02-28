"""
Based on DeepPoly_f from DiffAI (https://github.com/eth-sri/diffai/blob/master/ai.py)
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from src.abstract_layers.abstract_sig_base import SigBase
from src.utilities.general import tensor_reduce

if TYPE_CHECKING:
    from src.abstract_layers.abstract_bn2d import BatchNorm2d

from src.abstract_domains.ai_util import (
    AbstractElement,
    clamp_image,
    get_neg_pos_comp,
    head_from_bounds,
)
from src.abstract_domains.zonotope import HybridZonotope


class DeepPoly_f(AbstractElement):
    def __init__(
        self,
        inputs: HybridZonotope,
        x_l_coef: Tensor,
        x_u_coef: Tensor,
        x_l_bias: Optional[Tensor] = None,
        x_u_bias: Optional[Tensor] = None,
        input_error_map: Optional[Tensor] = None,
    ) -> None:
        super(DeepPoly_f, self).__init__()
        dtype = x_l_coef.dtype
        device = x_l_coef.device
        self.x_l_coef = x_l_coef
        self.x_u_coef = x_u_coef
        self.x_l_bias = (
            torch.zeros(x_l_coef.shape[1:], device=device, dtype=dtype)
            if x_l_bias is None
            else x_l_bias
        )
        self.x_u_bias = (
            torch.zeros(x_l_coef.shape[1:], device=device, dtype=dtype)
            if x_u_bias is None
            else x_u_bias
        )
        self.input_error_map = (
            input_error_map
            if input_error_map is not None
            else torch.arange(0, self.x_l_coef[0].numel())
        )
        assert self.input_error_map.shape[0] == self.x_l_coef.shape[0]

        self.inputs = inputs
        self.domain = "DPF"

    @classmethod
    def construct_from_noise(
        cls,
        x: Tensor,
        eps: Union[float, Tensor],
        data_range: Tuple[float, float] = (0, 1),
        dtype: Optional[torch.dtype] = None,
        domain: Optional[str] = None,
    ) -> "DeepPoly_f":
        # compute center and error terms from input, perturbation size and data range
        assert domain is None or domain == "DPF"
        if dtype is None:
            dtype = x.dtype
        if data_range is None:
            data_range = (-np.inf, np.inf)
        assert data_range[0] < data_range[1]
        x_center, x_beta = clamp_image(x, eps, data_range[0], data_range[1])
        x_center, x_beta = x_center.to(dtype=dtype), x_beta.to(dtype=dtype)
        return cls.construct(
            x_center, HybridZonotope.construct(x_center, x_beta, domain="box")
        )

    @classmethod
    def construct_constant(
        cls,
        x: Tensor,
        inputs: HybridZonotope,
        dtype: Optional[torch.dtype] = None,
        domain: Optional[str] = None,
    ) -> "DeepPoly_f":
        # compute center and error terms from input, perturbation size and data range
        assert domain is None or domain == "DPF"
        if dtype is None:
            dtype = x.dtype
        x_l_coef = torch.zeros(
            [inputs.head[0].numel() if inputs is not None else 1, *x.shape],
            dtype=dtype,
            device=x.device,
        )
        x_u_coef = torch.zeros(
            [inputs.head[0].numel() if inputs is not None else 1, *x.shape],
            dtype=dtype,
            device=x.device,
        )
        return cls(inputs, x_l_coef, x_u_coef, x, x)

    @classmethod
    def construct_from_bounds(
        cls,
        min_x: Tensor,
        max_x: Tensor,
        dtype: Optional[torch.dtype] = None,
        domain: Optional[str] = None,
    ) -> "DeepPoly_f":
        dtype = torch.get_default_dtype() if dtype is None else dtype
        assert domain is None or domain == "DPF"
        assert min_x.shape == max_x.shape
        x_center, x_beta = head_from_bounds(min_x, max_x)
        x_center, x_beta = x_center.to(dtype=dtype), x_beta.to(dtype=dtype)
        return cls.construct(
            x_center, HybridZonotope.construct(x_center, x_beta, domain="box")
        )

    @classmethod
    def construct_from_zono(cls, input_zono: HybridZonotope) -> "DeepPoly_f":
        assert input_zono.beta is None
        assert input_zono.errors is not None
        x_l_coef = input_zono.errors.clone()
        x_u_coef = input_zono.errors.clone()
        x_l_bias = input_zono.head.clone()
        x_u_bias = input_zono.head.clone()
        base_box = HybridZonotope.construct_from_noise(
            torch.zeros_like(input_zono.head), eps=1, domain="box", data_range=(-1, 1)
        )
        return cls(base_box, x_l_coef, x_u_coef, x_l_bias, x_u_bias)

    @staticmethod
    def construct(x: Tensor, inputs: "HybridZonotope") -> "DeepPoly_f":
        lb_in, ub_in = inputs.concretize()
        non_zero_width_dim = (
            (ub_in != lb_in).flatten(1).any(0)
        )  # ((ub_in - lb_in) > 0).flatten(1).any(0)
        k = int(non_zero_width_dim.sum().item())
        assert k > 0, "No nonzero dimension found"
        input_error_map = torch.arange(0, lb_in[0].numel())[non_zero_width_dim]
        x_l_coef = torch.zeros((x.shape[0], k * x[0].numel())).to(x.device)
        x_l_coef[
            :,
            torch.arange(k).to(lb_in.device) * x[0].numel()
            + non_zero_width_dim.nonzero()[:, 0],
        ] = 1.0
        x_l_coef = (
            x_l_coef.view(x.shape[0], k, -1)
            .permute(1, 0, 2)
            .view(k, *x.shape)
            .contiguous()
        )
        x_u_coef = x_l_coef.clone().detach()
        x_l_bias = torch.where(
            non_zero_width_dim.view(1, *x.shape[1:]), torch.zeros_like(lb_in), lb_in
        )
        x_u_bias = torch.where(
            non_zero_width_dim.view(1, *x.shape[1:]), torch.zeros_like(lb_in), ub_in
        )
        return DeepPoly_f(
            inputs=inputs.flatten()[:, non_zero_width_dim],  # type: ignore  # indexing with bool tensor
            x_l_coef=x_l_coef,
            x_u_coef=x_u_coef,
            x_l_bias=x_l_bias,
            x_u_bias=x_u_bias,
            input_error_map=input_error_map,
        )

    def dim(self) -> int:
        return self.x_l_coef.dim() - 1

    @staticmethod
    def cat(x: List["DeepPoly_f"], dim: int = 0) -> "DeepPoly_f":  # type: ignore [override]
        assert all([x[0].inputs == y.inputs for y in x])

        actual_dim = dim if dim >= 0 else x[0].dim() + dim
        assert 0 <= actual_dim < x[0].dim()

        x_l_coef = torch.cat([x_i.x_l_coef for x_i in x], dim=actual_dim + 1)
        x_u_coef = torch.cat([x_i.x_u_coef for x_i in x], dim=actual_dim + 1)
        x_l_bias = torch.cat([x_i.x_l_bias for x_i in x], dim=actual_dim)
        x_u_bias = torch.cat([x_i.x_u_bias for x_i in x], dim=actual_dim)

        return DeepPoly_f(
            x[0].inputs, x_l_coef, x_u_coef, x_l_bias, x_u_bias, x[0].input_error_map
        )

    @staticmethod
    def stack(x: List["DeepPoly_f"], dim: int = 0) -> "DeepPoly_f":
        assert all([x[0].inputs == y.inputs for y in x])

        actual_dim = dim if dim >= 0 else x[0].dim() + dim + 1
        assert 0 <= actual_dim <= x[0].dim()

        x_l_coef = torch.stack([x_i.x_l_coef for x_i in x], dim=actual_dim + 1)
        x_u_coef = torch.stack([x_i.x_u_coef for x_i in x], dim=actual_dim + 1)
        x_l_bias = torch.stack([x_i.x_l_bias for x_i in x], dim=actual_dim)
        x_u_bias = torch.stack([x_i.x_u_bias for x_i in x], dim=actual_dim)

        return DeepPoly_f(
            x[0].inputs, x_l_coef, x_u_coef, x_l_bias, x_u_bias, x[0].input_error_map
        )

    def size(self, idx: Optional[int] = None) -> Union[Tuple[int, ...], int]:
        if idx is None:
            return self.shape
        else:
            return self.shape[idx]

    def view(self, size: Tuple[int, ...]) -> "DeepPoly_f":
        input_terms = self.x_l_coef.shape[0]
        return DeepPoly_f(
            self.inputs,
            self.x_l_coef.view(input_terms, *size),
            self.x_u_coef.view(input_terms, *size),
            self.x_l_bias.view(*size),
            self.x_u_bias.view(*size),
            self.input_error_map,
        )

    @property
    def shape(self) -> torch.Size:
        return self.x_l_bias.shape

    @property
    def device(self) -> torch.device:
        return self.x_l_bias.device

    @property
    def dtype(self) -> torch.dtype:
        return self.x_l_bias.dtype

    def flatten(self) -> "DeepPoly_f":
        return self.view((*self.shape[:1], -1))

    def normalize(self, mean: Tensor, sigma: Tensor) -> "DeepPoly_f":
        return (self - mean) / sigma

    def __sub__(  # type: ignore[override]
        self, other: Union[Tensor, float, int, DeepPoly_f]
    ) -> "DeepPoly_f":
        if (
            isinstance(other, torch.Tensor)
            or isinstance(other, float)
            or isinstance(other, int)
        ):
            return DeepPoly_f(
                self.inputs,
                self.x_l_coef,
                self.x_u_coef,
                self.x_l_bias - other,
                self.x_u_bias - other,
                self.input_error_map,
            )
        elif isinstance(other, DeepPoly_f):
            assert self.inputs == other.inputs
            return DeepPoly_f(
                self.inputs,
                self.x_l_coef - other.x_u_coef,
                self.x_u_coef - other.x_l_coef,
                self.x_l_bias - other.x_u_bias,
                self.x_u_bias - other.x_l_bias,
                self.input_error_map,
            )
        else:
            assert False, "Unknown type of other object"

    def __neg__(self) -> "DeepPoly_f":
        return DeepPoly_f(
            self.inputs,
            -self.x_u_coef,
            -self.x_l_coef,
            -self.x_u_bias,
            -self.x_l_bias,
            self.input_error_map,
        )

    def __add__(  # type: ignore[override]
        self, other: Union[Tensor, float, int, DeepPoly_f]
    ) -> "DeepPoly_f":
        if (
            isinstance(other, torch.Tensor)
            or isinstance(other, float)
            or isinstance(other, int)
        ):
            return DeepPoly_f(
                self.inputs,
                self.x_l_coef,
                self.x_u_coef,
                self.x_l_bias + other,
                self.x_u_bias + other,
                self.input_error_map,
            )
        elif isinstance(other, DeepPoly_f):
            assert self.inputs == other.inputs
            return DeepPoly_f(
                self.inputs,
                self.x_l_coef + other.x_l_coef,
                self.x_u_coef + other.x_u_coef,
                self.x_l_bias + other.x_l_bias,
                self.x_u_bias + other.x_u_bias,
                self.input_error_map,
            )
        else:
            assert False, "Unknown type of other object"

    def __truediv__(self, other: Union[Tensor, float, int]) -> "DeepPoly_f":
        if isinstance(other, torch.Tensor):
            assert (other != 0).all()
            x_l_coef = torch.where(
                other >= 0, self.x_l_coef / other, self.x_u_coef / other
            )
            x_u_coef = torch.where(
                other >= 0, self.x_u_coef / other, self.x_l_coef / other
            )
            x_l_bias = torch.where(
                other >= 0, self.x_l_bias / other, self.x_u_bias / other
            )
            x_u_bias = torch.where(
                other >= 0, self.x_u_bias / other, self.x_l_bias / other
            )
        elif isinstance(other, float) or isinstance(other, int):
            assert other != 0
            x_l_coef = self.x_l_coef / other if other >= 0 else self.x_u_coef / other
            x_u_coef = self.x_u_coef / other if other >= 0 else self.x_l_coef / other
            x_l_bias = self.x_l_bias / other if other >= 0 else self.x_u_bias / other
            x_u_bias = self.x_u_bias / other if other >= 0 else self.x_l_bias / other
        else:
            assert False, "Unknown type of other object"
        return DeepPoly_f(
            self.inputs, x_l_coef, x_u_coef, x_l_bias, x_u_bias, self.input_error_map
        )

    def __rmul__(  # type: ignore # Complains about unsafe overlap with __mul__
        self, other: Union[Tensor, float, int]
    ) -> "DeepPoly_f":
        return self.__mul__(other)

    def __mul__(self, other: Union[Tensor, float, int]) -> "DeepPoly_f":
        if isinstance(other, torch.Tensor):
            x_l_coef = torch.where(
                other >= 0, self.x_l_coef * other, self.x_u_coef * other
            )
            x_u_coef = torch.where(
                other >= 0, self.x_u_coef * other, self.x_l_coef * other
            )
            x_l_bias = torch.where(
                other >= 0, self.x_l_bias * other, self.x_u_bias * other
            )
            x_u_bias = torch.where(
                other >= 0, self.x_u_bias * other, self.x_l_bias * other
            )
            return DeepPoly_f(
                self.inputs,
                x_l_coef,
                x_u_coef,
                x_l_bias,
                x_u_bias,
                self.input_error_map,
            )
        elif isinstance(other, int) or isinstance(other, float):
            x_l_coef = self.x_l_coef * other if other >= 0 else self.x_u_coef * other
            x_u_coef = self.x_u_coef * other if other >= 0 else self.x_l_coef * other
            x_l_bias = self.x_l_bias * other if other >= 0 else self.x_u_bias * other
            x_u_bias = self.x_u_bias * other if other >= 0 else self.x_l_bias * other
            return DeepPoly_f(
                self.inputs,
                x_l_coef,
                x_u_coef,
                x_l_bias,
                x_u_bias,
                self.input_error_map,
            )
        else:
            assert False, "Unknown type of other object"

    def __getitem__(self, indices: Tuple[int, ...]) -> "DeepPoly_f":
        if not isinstance(indices, tuple):
            indices = tuple([indices])
        return DeepPoly_f(
            self.inputs,
            self.x_l_coef[(slice(None, None, None), *indices)],
            self.x_u_coef[(slice(None, None, None), *indices)],
            self.x_l_bias[indices],
            self.x_u_bias[indices],
            self.input_error_map,
        )

    def clone(self) -> "DeepPoly_f":
        return DeepPoly_f(
            self.inputs,
            self.x_l_coef.clone(),
            self.x_u_coef.clone(),
            self.x_l_bias.clone(),
            self.x_u_bias.clone(),
            self.input_error_map.clone(),
        )

    def detach(self) -> "DeepPoly_f":
        return DeepPoly_f(
            self.inputs,
            self.x_l_coef.detach(),
            self.x_u_coef.detach(),
            self.x_l_bias.detach(),
            self.x_u_bias.detach(),
            self.input_error_map.detach(),
        )

    def max_center(self) -> Tensor:
        return self.x_u_bias.max(dim=1)[0].unsqueeze(1)

    def max_pool2d(
        self,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int],
        padding: Tuple[int, int],
    ) -> "DeepPoly_f":
        k = self.x_l_coef.shape[0]
        in_lb, in_ub = self.concretize()
        device = in_lb.device

        pid_lb = F.pad(
            in_lb,
            (padding[1], padding[1], padding[0], padding[0]),
            value=-torch.inf,
        )
        pid_ub = F.pad(
            in_ub,
            (padding[1], padding[1], padding[0], padding[0]),
            value=-torch.inf,
        )

        output_ub = F.max_pool2d(in_ub, kernel_size, stride, padding)
        output_lb = F.max_pool2d(in_lb, kernel_size, stride, padding)
        tight = (output_ub == output_lb).all(0).all(0)
        output_dim = output_ub.shape

        output_lb, output_ub = output_lb.flatten(), output_ub.flatten()

        input_dim = in_ub.shape

        x_l_coef = torch.zeros((k, np.prod(output_dim)), device=device)
        x_u_coef = torch.zeros((k, np.prod(output_dim)), device=device)
        x_u_bias = output_ub.clone().flatten()
        x_l_bias = output_lb.clone().flatten()

        self_x_l_coef = self.x_l_coef.flatten(1)
        self_x_u_coef = self.x_u_coef.flatten(1)
        self_x_l_bias = self.x_l_bias.flatten()
        self_x_u_bias = self.x_u_bias.flatten()

        offsets_in = torch.tensor(
            [int(np.prod(input_dim[i + 1 :])) for i in range(len(input_dim))],
            device=device,
        )
        offsets_out = torch.tensor(
            [int(np.prod(output_dim[i + 1 :])) for i in range(len(output_dim))],
            device=device,
        )

        ch_range = torch.arange(output_dim[1], device=device).repeat(output_dim[0])
        bs_range = torch.arange(output_dim[0], device=device).repeat_interleave(
            output_dim[1]
        )
        for y in torch.arange(output_dim[2])[~tight.all(1)]:
            for x in torch.arange(output_dim[3])[~tight[y]]:
                if tight[y, x]:
                    assert False

                # Get the input_window
                w_in_idy = y * stride[0]
                w_in_idx = x * stride[1]
                w_lb = pid_lb[
                    :,
                    :,
                    w_in_idy : w_in_idy + kernel_size[0],
                    w_in_idx : w_in_idx + kernel_size[1],
                ].flatten(start_dim=2)
                w_ub = pid_ub[
                    :,
                    :,
                    w_in_idy : w_in_idy + kernel_size[0],
                    w_in_idx : w_in_idx + kernel_size[1],
                ].flatten(start_dim=2)

                best_lb, best_lb_i = w_lb.max(2)
                best_lb_i = best_lb_i.view(-1)
                max_ub = w_ub.max(2)[0]
                strict_dom = (
                    torch.sum((best_lb.unsqueeze(2) <= w_ub).float(), 2) == 1.0
                ).view(-1)

                in_idx = best_lb_i % kernel_size[1]
                in_idy = torch.div(best_lb_i, kernel_size[1], rounding_mode="trunc")
                tot_idx = in_idx + w_in_idx - padding[0]
                tot_idy = in_idy + w_in_idy - padding[1]
                tot_idx_valid = (
                    (tot_idx >= 0)
                    & (tot_idx < input_dim[3])
                    & (tot_idy >= 0)
                    & (tot_idy < input_dim[2])
                )
                assert all(tot_idx_valid)

                in_idx = (
                    bs_range * offsets_in[0]
                    + ch_range * offsets_in[1]
                    + tot_idy * offsets_in[2]
                    + tot_idx * offsets_in[3]
                )
                out_idx = (
                    bs_range * offsets_out[0]
                    + ch_range * offsets_out[1]
                    + y * offsets_out[2]
                    + x * offsets_out[3]
                )

                assert (max_ub.flatten() == output_ub[out_idx]).all()

                x_u_coef[:, out_idx[strict_dom]] = self_x_u_coef[:, in_idx[strict_dom]]
                x_u_bias[out_idx[strict_dom]] = self_x_u_bias[in_idx[strict_dom]]

                x_l_coef[:, out_idx] = self_x_l_coef[:, in_idx]
                x_l_bias[out_idx] = self_x_l_bias[in_idx]

                # x_u_coef[:, bs_range, ch_range, y, x] = torch.where((strict_dom & tot_idx_valid).unsqueeze(0), self.x_u_coef[:,bs_range,ch_range,tot_idy,tot_idx], x_u_coef[:, bs_range, ch_range, y, x])
                # x_u_bias[bs_range, ch_range, y, x] = torch.where(strict_dom & tot_idx_valid, self.x_u_bias[bs_range, ch_range, tot_idy, tot_idx], output_ub[:, ch_range, y, x])
                #
                # x_l_coef[:, bs_range, ch_range, y, x] = torch.where(tot_idx_valid, self.x_l_coef[:,bs_range,ch_range,tot_idy,tot_idx], x_l_coef[:, bs_range, ch_range, y, x])
                # x_l_bias[bs_range, ch_range, y, x] = torch.where(tot_idx_valid, self.x_l_bias[:, ch_range, tot_idy, tot_idx], output_lb[:, ch_range, y, x])

        x_u_coef = x_u_coef.view(k, *output_dim)
        x_l_coef = x_l_coef.view(k, *output_dim)
        x_u_bias = x_u_bias.view(*output_dim)
        x_l_bias = x_l_bias.view(*output_dim)

        return DeepPoly_f(
            self.inputs, x_l_coef, x_u_coef, x_l_bias, x_u_bias, self.input_error_map
        )

    def avg_pool2d(self, kernel_size: int, stride: int, padding: int) -> "DeepPoly_f":
        n_in_dims = self.x_l_coef.shape[0:2]

        x_l_coef = F.avg_pool2d(
            self.x_l_coef.view(-1, *self.x_l_coef.shape[2:]),
            kernel_size,
            stride,
            padding,
        )
        x_l_coef = x_l_coef.view(*n_in_dims, *x_l_coef.shape[1:])
        x_u_coef = F.avg_pool2d(
            self.x_u_coef.view(-1, *self.x_l_coef.shape[2:]),
            kernel_size,
            stride,
            padding,
        )
        x_u_coef = x_u_coef.view(*n_in_dims, *x_u_coef.shape[1:])

        x_l_bias = F.avg_pool2d(self.x_l_bias, kernel_size, stride, padding)
        x_u_bias = F.avg_pool2d(self.x_u_bias, kernel_size, stride, padding)
        return DeepPoly_f(
            self.inputs, x_l_coef, x_u_coef, x_l_bias, x_u_bias, self.input_error_map
        )

    def global_avg_pool2d(self) -> "DeepPoly_f":
        n_in_dims = self.x_l_coef.shape[0:2]

        x_l_coef = F.adaptive_avg_pool2d(
            self.x_l_coef.view(-1, *self.x_l_coef.shape[2:]), 1
        )
        x_l_coef = x_l_coef.view(*n_in_dims, *x_l_coef.shape[1:])
        x_u_coef = F.adaptive_avg_pool2d(
            self.x_u_coef.view(-1, *self.x_l_coef.shape[2:]), 1
        )
        x_u_coef = x_u_coef.view(*n_in_dims, *x_u_coef.shape[1:])

        x_l_bias = F.adaptive_avg_pool2d(self.x_l_bias, 1)
        x_u_bias = F.adaptive_avg_pool2d(self.x_u_bias, 1)
        return DeepPoly_f(
            self.inputs, x_l_coef, x_u_coef, x_l_bias, x_u_bias, self.input_error_map
        )

    def pad(
        self, pad: Tuple[int, ...], mode: str = "constant", value: float = 0.0
    ) -> "DeepPoly_f":
        assert mode == "constant"
        x_l_bias = F.pad(self.x_l_bias, pad, mode="constant", value=value)
        x_u_bias = F.pad(self.x_u_bias, pad, mode="constant", value=value)
        x_l_coef = F.pad(
            self.x_l_coef.view(-1, *self.x_l_coef.shape[2:]),
            pad,
            mode="constant",
            value=0,
        ).view(-1, *x_l_bias.shape)
        x_u_coef = F.pad(
            self.x_u_coef.view(-1, *self.x_u_coef.shape[2:]),
            pad,
            mode="constant",
            value=0,
        ).view(-1, *x_u_bias.shape)
        return DeepPoly_f(
            self.inputs, x_l_coef, x_u_coef, x_l_bias, x_u_bias, self.input_error_map
        )

    def conv2d(
        self,
        weight: Tensor,
        bias: Optional[Tensor],
        stride: int,
        padding: int,
        dilation: int,
        groups: int,
    ) -> "DeepPoly_f":
        n_in_dims = self.x_l_coef.shape[0:2]

        weight_neg, weight_pos = get_neg_pos_comp(weight)

        x_l_coef = F.conv2d(
            self.x_l_coef.view(-1, *self.x_l_coef.shape[2:]),
            weight_pos,
            None,
            stride,
            padding,
            dilation,
            groups,
        ) + F.conv2d(
            self.x_u_coef.view(-1, *self.x_l_coef.shape[2:]),
            weight_neg,
            None,
            stride,
            padding,
            dilation,
            groups,
        )
        x_l_coef = x_l_coef.view(*n_in_dims, *x_l_coef.shape[1:])
        x_u_coef = F.conv2d(
            self.x_u_coef.view(-1, *self.x_l_coef.shape[2:]),
            weight_pos,
            None,
            stride,
            padding,
            dilation,
            groups,
        ) + F.conv2d(
            self.x_l_coef.view(-1, *self.x_l_coef.shape[2:]),
            weight_neg,
            None,
            stride,
            padding,
            dilation,
            groups,
        )
        x_u_coef = x_u_coef.view(*n_in_dims, *x_u_coef.shape[1:])

        x_l_bias = F.conv2d(
            self.x_l_bias, weight_pos, bias, stride, padding, dilation, groups
        ) + F.conv2d(self.x_u_bias, weight_neg, None, stride, padding, dilation, groups)
        x_u_bias = F.conv2d(
            self.x_u_bias, weight_pos, bias, stride, padding, dilation, groups
        ) + F.conv2d(self.x_l_bias, weight_neg, None, stride, padding, dilation, groups)
        return DeepPoly_f(
            self.inputs, x_l_coef, x_u_coef, x_l_bias, x_u_bias, self.input_error_map
        )

    def convtranspose2d(
        self,
        weight: Tensor,
        bias: Optional[Tensor],
        stride: int,
        padding: int,
        out_padding: int,
        groups: int,
        dilation: int,
    ) -> "DeepPoly_f":
        n_in_dims = self.x_l_coef.shape[0:2]

        weight_neg, weight_pos = get_neg_pos_comp(weight)

        x_l_coef = F.conv_transpose2d(
            self.x_l_coef.view(-1, *self.x_l_coef.shape[2:]),
            weight_pos,
            None,
            stride,
            padding,
            out_padding,
            groups,
            dilation,
        ) + F.conv_transpose2d(
            self.x_u_coef.view(-1, *self.x_l_coef.shape[2:]),
            weight_neg,
            None,
            stride,
            padding,
            out_padding,
            groups,
            dilation,
        )
        x_l_coef = x_l_coef.view(*n_in_dims, *x_l_coef.shape[1:])
        x_u_coef = F.conv_transpose2d(
            self.x_u_coef.view(-1, *self.x_l_coef.shape[2:]),
            weight_pos,
            None,
            stride,
            padding,
            out_padding,
            groups,
            dilation,
        ) + F.conv_transpose2d(
            self.x_l_coef.view(-1, *self.x_l_coef.shape[2:]),
            weight_neg,
            None,
            stride,
            padding,
            out_padding,
            groups,
            dilation,
        )
        x_u_coef = x_u_coef.view(*n_in_dims, *x_u_coef.shape[1:])

        x_l_bias = F.conv_transpose2d(
            self.x_l_bias,
            weight_pos,
            bias,
            stride,
            padding,
            out_padding,
            groups,
            dilation,
        ) + F.conv_transpose2d(
            self.x_u_bias,
            weight_neg,
            None,
            stride,
            padding,
            out_padding,
            groups,
            dilation,
        )
        x_u_bias = F.conv_transpose2d(
            self.x_u_bias,
            weight_pos,
            bias,
            stride,
            padding,
            out_padding,
            groups,
            dilation,
        ) + F.conv_transpose2d(
            self.x_l_bias,
            weight_neg,
            None,
            stride,
            padding,
            out_padding,
            groups,
            dilation,
        )
        return DeepPoly_f(
            self.inputs, x_l_coef, x_u_coef, x_l_bias, x_u_bias, self.input_error_map
        )

    def linear(
        self,
        weight: Tensor,
        bias: Union[Tensor, None] = None,
        C: Union[Tensor, None] = None,
    ) -> "DeepPoly_f":
        n_in_dims = self.x_l_coef.shape[0:2]

        weight_neg, weight_pos = get_neg_pos_comp(weight)

        x_l_coef = F.linear(
            self.x_l_coef.view(-1, *self.x_l_coef.shape[2:]), weight_pos, None
        ) + F.linear(self.x_u_coef.view(-1, *self.x_l_coef.shape[2:]), weight_neg, None)
        x_l_coef = x_l_coef.view(*n_in_dims, *x_l_coef.shape[1:])
        x_u_coef = F.linear(
            self.x_u_coef.view(-1, *self.x_l_coef.shape[2:]), weight_pos, None
        ) + F.linear(self.x_l_coef.view(-1, *self.x_l_coef.shape[2:]), weight_neg, None)
        x_u_coef = x_u_coef.view(*n_in_dims, *x_u_coef.shape[1:])

        x_l_bias = F.linear(self.x_l_bias, weight_pos, bias) + F.linear(
            self.x_u_bias, weight_neg, None
        )
        x_u_bias = F.linear(self.x_u_bias, weight_pos, bias) + F.linear(
            self.x_l_bias, weight_neg, None
        )
        return DeepPoly_f(
            self.inputs, x_l_coef, x_u_coef, x_l_bias, x_u_bias, self.input_error_map
        )

    def matmul(self, other: Tensor) -> "DeepPoly_f":
        n_in_dims = self.x_l_coef.shape[0:2]

        weight_neg, weight_pos = get_neg_pos_comp(other)

        x_l_coef = torch.matmul(
            self.x_l_coef.view(-1, *self.x_l_coef.shape[2:]), weight_pos
        ) + torch.matmul(self.x_u_coef.view(-1, *self.x_l_coef.shape[2:]), weight_neg)
        x_l_coef = x_l_coef.view(*n_in_dims, *x_l_coef.shape[1:])
        x_u_coef = torch.matmul(
            self.x_u_coef.view(-1, *self.x_l_coef.shape[2:]), weight_pos
        ) + torch.matmul(self.x_l_coef.view(-1, *self.x_l_coef.shape[2:]), weight_neg)
        x_u_coef = x_u_coef.view(*n_in_dims, *x_u_coef.shape[1:])

        x_l_bias = torch.matmul(self.x_l_bias, weight_pos) + torch.matmul(
            self.x_u_bias, weight_neg
        )
        x_u_bias = torch.matmul(self.x_u_bias, weight_pos) + torch.matmul(
            self.x_l_bias, weight_neg
        )
        return DeepPoly_f(
            self.inputs, x_l_coef, x_u_coef, x_l_bias, x_u_bias, self.input_error_map
        )

    def einsum(self, defining_str: str, other: Tensor) -> DeepPoly_f:
        input_self_str, rest = defining_str.split(",")
        input_other_str, output_str = rest.split("->")
        input_self_str, input_other_str, output_str = (
            input_self_str.strip(),
            input_other_str.strip(),
            output_str.strip(),
        )

        weight_neg, weight_pos = get_neg_pos_comp(other)
        x_l_coef = torch.einsum(
            f"i{input_self_str},{input_other_str} -> i{output_str}",
            self.x_l_coef,
            weight_pos,
        ) + torch.einsum(
            f"i{input_self_str},{input_other_str} -> i{output_str}",
            self.x_u_coef,
            weight_neg,
        )
        x_u_coef = torch.einsum(
            f"i{input_self_str},{input_other_str} -> i{output_str}",
            self.x_u_coef,
            weight_pos,
        ) + torch.einsum(
            f"i{input_self_str},{input_other_str} -> i{output_str}",
            self.x_l_coef,
            weight_neg,
        )
        x_l_bias = torch.einsum(defining_str, self.x_l_bias, weight_pos) + torch.einsum(
            defining_str, self.x_u_bias, weight_neg
        )
        x_u_bias = torch.einsum(defining_str, self.x_u_bias, weight_pos) + torch.einsum(
            defining_str, self.x_l_bias, weight_neg
        )

        return DeepPoly_f(
            self.inputs, x_l_coef, x_u_coef, x_l_bias, x_u_bias, self.input_error_map
        )

    def rev_matmul(self, other: Tensor) -> "DeepPoly_f":
        n_in_dims = self.x_l_coef.shape[0:2]

        weight_neg, weight_pos = get_neg_pos_comp(other)

        x_l_coef = torch.matmul(
            weight_pos, self.x_l_coef.view(-1, *self.x_l_coef.shape[2:])
        ) + torch.matmul(weight_neg, self.x_u_coef.view(-1, *self.x_l_coef.shape[2:]))
        x_l_coef = x_l_coef.view(*n_in_dims, *x_l_coef.shape[1:])
        x_u_coef = torch.matmul(
            weight_pos, self.x_u_coef.view(-1, *self.x_l_coef.shape[2:])
        ) + torch.matmul(weight_neg, self.x_l_coef.view(-1, *self.x_l_coef.shape[2:]))
        x_u_coef = x_u_coef.view(*n_in_dims, *x_u_coef.shape[1:])

        x_l_bias = torch.matmul(weight_pos, self.x_l_bias) + torch.matmul(
            weight_neg, self.x_u_bias
        )
        x_u_bias = torch.matmul(weight_pos, self.x_u_bias) + torch.matmul(
            weight_neg, self.x_l_bias
        )
        return DeepPoly_f(
            self.inputs, x_l_coef, x_u_coef, x_l_bias, x_u_bias, self.input_error_map
        )

    def batch_norm(self, bn: BatchNorm2d) -> "DeepPoly_f":
        view_dim_list = [1, -1] + (self.x_l_bias.dim() - 2) * [1]
        self_stat_dim_list = [0, 2, 3] if self.x_l_bias.dim() == 4 else [0]
        if bn.training and (bn.current_var is None or bn.current_mean is None):
            if bn.running_mean is not None and bn.running_var is not None:
                bn.current_mean = bn.running_mean
                bn.current_var = bn.running_var
            else:
                bn.current_mean = (
                    (0.5 * (self.x_l_bias + self.x_u_bias))
                    .mean(dim=self_stat_dim_list)
                    .detach()
                )
                bn.current_var = (
                    (0.5 * (self.x_l_bias + self.x_u_bias))
                    .var(unbiased=False, dim=self_stat_dim_list)
                    .detach()
                )

        c: Tensor = bn.weight / torch.sqrt(
            bn.current_var + bn.eps
        )  # type: ignore # Type inference does not work for the module attributes
        b: Tensor = -bn.current_mean * c + bn.bias  # type: ignore  # type: ignore # Type inference does not work for the module attributes

        out_dp = self * c.view(*view_dim_list) + b.view(*view_dim_list)
        return out_dp

    def relu(
        self,
        deepz_lambda: Optional[Tensor] = None,
        bounds: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple["DeepPoly_f", Optional[Tensor]]:
        lb, ub = self.concretize()
        init_lambda = False
        assert (ub - lb >= 0).all(), f"max violation: {(ub - lb).min()}"

        dtype = lb.dtype
        D = 1e-8

        if bounds is not None:
            lb_refined, ub_refined = bounds
            lb = torch.max(lb_refined, lb)
            ub = torch.min(ub_refined, ub)

        is_cross = (lb < 0) & (ub > 0)

        lambda_u = torch.where(is_cross, ub / (ub - lb + D), (lb >= 0).to(dtype))
        lambda_l = torch.where(ub < -lb, torch.zeros_like(lb), torch.ones_like(lb))
        lambda_l = torch.where(is_cross, lambda_l, (lb >= 0).to(dtype))

        # lambda_l = torch.where(is_cross, lambda_u, (lb >= 0).to(dtype))

        if deepz_lambda is not None:
            if (
                (deepz_lambda >= 0).all() and (deepz_lambda <= 1).all()
            ) and not init_lambda:
                lambda_l = deepz_lambda
            else:
                deepz_lambda.data = lambda_l.data.detach().requires_grad_(True)

        mu_l = torch.zeros_like(lb)
        mu_u = torch.where(
            is_cross, -lb * lambda_u, torch.zeros_like(lb)
        )  # height of upper bound intersection with y axis

        x_l_bias = mu_l + lambda_l * self.x_l_bias
        x_u_bias = mu_u + lambda_u * self.x_u_bias
        lambda_l, lambda_u = lambda_l.unsqueeze(0), lambda_u.unsqueeze(0)
        x_l_coef = self.x_l_coef * lambda_l
        x_u_coef = self.x_u_coef * lambda_u

        DP_out = DeepPoly_f(
            self.inputs, x_l_coef, x_u_coef, x_l_bias, x_u_bias, self.input_error_map
        )

        assert (DP_out.concretize()[1] - DP_out.concretize()[0] >= 0).all()
        # return DeepPoly_f(self.inputs, x_l_coef, x_u_coef, x_l_bias, x_u_bias), deepz_lambda
        return DP_out, deepz_lambda

    def sigmoid(
        self,
        step_size: float,
        max_x: float,
        deepz_lambda: Optional[Tensor] = None,
        bounds: Optional[Tuple[Tensor, Tensor]] = None,
        tangent_points: Optional[Tensor] = None,
    ) -> Tuple["DeepPoly_f", Optional[Tensor]]:
        def sig(x: Tensor) -> Tensor:
            return torch.sigmoid(x)

        def d_sig(x: Tensor) -> Tensor:
            sig = torch.sigmoid(x)
            return sig * (1 - sig)

        assert tangent_points is not None, "Tangent points not set"
        if tangent_points.device != self.device:
            tangent_points = tangent_points.to(device=self.device)
        if tangent_points.dtype != self.dtype:
            tangent_points = tangent_points.to(dtype=self.dtype)

        lb, ub = self.concretize()
        assert (ub - lb >= 0).all(), f"max violation: {(ub - lb).min()}"

        if bounds is not None:
            lb_refined, ub_refined = bounds
            lb = torch.max(lb_refined, lb)
            ub = torch.min(ub_refined, ub)

        (
            lb_slope,
            ub_slope,
            lb_intercept,
            ub_intercept,
        ) = SigBase._get_approximation_slopes_and_intercepts_for_act(
            (lb, ub),
            tangent_points,
            step_size,
            max_x,
            sig,
            d_sig,
        )

        x_l_bias = lb_intercept + lb_slope * self.x_l_bias
        x_u_bias = ub_intercept + ub_slope * self.x_u_bias
        lambda_l, lambda_u = lb_slope.unsqueeze(0), ub_slope.unsqueeze(0)
        x_l_coef = self.x_l_coef * lambda_l
        x_u_coef = self.x_u_coef * lambda_u

        DP_out = DeepPoly_f(
            self.inputs, x_l_coef, x_u_coef, x_l_bias, x_u_bias, self.input_error_map
        )

        assert (DP_out.concretize()[1] - DP_out.concretize()[0] >= 0).all()
        return DP_out, deepz_lambda

    def unsqueeze(self, dim: int) -> "DeepPoly_f":
        return DeepPoly_f(
            self.inputs,
            self.x_l_coef.unsqueeze(dim + 1),
            self.x_u_coef.unsqueeze(dim + 1),
            self.x_l_bias.unsqueeze(dim),
            self.x_u_bias.unsqueeze(dim),
            self.input_error_map,
        )

    def squeeze(self, dim: int) -> "DeepPoly_f":
        return DeepPoly_f(
            self.inputs,
            self.x_l_coef.squeeze(dim + 1),
            self.x_u_coef.squeeze(dim + 1),
            self.x_l_bias.squeeze(dim),
            self.x_u_bias.squeeze(dim),
            self.input_error_map,
        )

    def split(
        self, split_size_or_sections: Tuple[int, ...], dim: int
    ) -> Tuple[DeepPoly_f, ...]:
        real_dim = dim if dim > 0 else self.x_l_bias.dim() + dim

        new_x_l_coef = torch.split(self.x_l_coef, split_size_or_sections, real_dim + 1)
        new_x_u_coef = torch.split(self.x_u_coef, split_size_or_sections, real_dim + 1)
        new_x_l_bias = torch.split(self.x_l_bias, split_size_or_sections, real_dim)
        new_x_u_bias = torch.split(self.x_u_bias, split_size_or_sections, real_dim)

        outputs = [
            DeepPoly_f(
                self.inputs,
                xlc,
                xuc,
                xlb,
                xub,
                self.input_error_map,
            )
            for xlc, xuc, xlb, xub in zip(
                new_x_l_coef, new_x_u_coef, new_x_l_bias, new_x_u_bias
            )
        ]

        return tuple(outputs)

    def slice(
        self,
        dim: int,
        starts: int,
        ends: int,
        steps: int,
    ) -> DeepPoly_f:
        real_dim = dim if dim > 0 else self.x_l_bias.dim() + dim

        index = torch.tensor(range(starts, ends, steps), device=self.x_l_bias.device)

        new_x_l_coef = torch.index_select(self.x_l_coef, real_dim + 1, index)
        new_x_u_coef = torch.index_select(self.x_u_coef, real_dim + 1, index)
        new_x_l_bias = torch.index_select(self.x_l_bias, real_dim, index)
        new_x_u_bias = torch.index_select(self.x_u_bias, real_dim, index)

        return DeepPoly_f(
            self.inputs,
            new_x_l_coef,
            new_x_u_coef,
            new_x_l_bias,
            new_x_u_bias,
            self.input_error_map,
        )

    def multiply_interval(self, interval: Tuple[Tensor, Tensor]) -> DeepPoly_f:

        concrete_lb, concrete_ub = self.concretize()

        (
            mul_lb_slope,
            mul_lb_intercept,
            mul_ub_slope,
            mul_ub_intercept,
        ) = self._get_multiplication_slopes_and_intercepts(
            interval, (concrete_lb, concrete_ub)
        )

        new_x_l_bias = mul_lb_intercept + mul_lb_slope * self.x_l_bias
        new_x_u_bias = mul_ub_intercept + mul_ub_slope * self.x_u_bias
        mul_lb_slope, mul_ub_slope = mul_lb_slope.unsqueeze(0), mul_ub_slope.unsqueeze(
            0
        )
        new_x_l_coef = self.x_l_coef * mul_lb_slope
        new_x_u_coef = self.x_u_coef * mul_ub_slope

        return DeepPoly_f(
            self.inputs,
            new_x_l_coef,
            new_x_u_coef,
            new_x_l_bias,
            new_x_u_bias,
            self.input_error_map,
        )

    def add(self, other: "DeepPoly_f") -> "DeepPoly_f":
        x_l_coef = self.x_l_coef + other.x_l_coef
        x_u_coef = self.x_u_coef + other.x_u_coef
        x_l_bias = self.x_l_bias + other.x_l_bias
        x_u_bias = self.x_u_bias + other.x_u_bias
        return DeepPoly_f(
            self.inputs, x_l_coef, x_u_coef, x_l_bias, x_u_bias, self.input_error_map
        )

    def sum(self, dim: int, reduce_dim: bool = False) -> "DeepPoly_f":
        x_l_coef = self.x_l_coef.sum(dim + 1)
        x_u_coef = self.x_u_coef.sum(dim + 1)
        x_l_bias = self.x_l_bias.sum(dim)
        x_u_bias = self.x_u_bias.sum(dim)

        if not reduce_dim:
            x_l_coef = x_l_coef.unsqueeze(dim + 1)
            x_l_coef = x_l_coef.unsqueeze(dim + 1)
            x_l_coef = x_l_coef.unsqueeze(dim)
            x_l_coef = x_l_coef.unsqueeze(dim)
        return DeepPoly_f(
            self.inputs, x_l_coef, x_u_coef, x_l_bias, x_u_bias, self.input_error_map
        )

    def concretize(self) -> Tuple[Tensor, Tensor]:
        input_lb, input_ub = self.inputs.concretize()
        input_shape = input_lb.shape
        input_lb = (
            input_lb.flatten(start_dim=1)
            .transpose(1, 0)
            .view([-1, input_shape[0]] + (self.x_l_coef.dim() - 2) * [1])
        )
        input_ub = (
            input_ub.flatten(start_dim=1)
            .transpose(1, 0)
            .view([-1, input_shape[0]] + (self.x_l_coef.dim() - 2) * [1])
        )
        neg_x_l_coef, pos_x_l_coef = get_neg_pos_comp(self.x_l_coef)
        neg_x_u_coef, pos_x_u_coef = get_neg_pos_comp(self.x_u_coef)

        lb = (
            self.x_l_bias
            + (neg_x_l_coef * input_ub).sum(0)
            + (pos_x_l_coef * input_lb).sum(0)
        )
        ub = (
            self.x_u_bias
            + (neg_x_u_coef * input_lb).sum(0)
            + (pos_x_u_coef * input_ub).sum(0)
        )
        return lb, ub

    def avg_width(self) -> Tensor:
        lb, ub = self.concretize()
        return (ub - lb).mean()

    def is_greater(
        self, i: int, j: int, threshold_min: Union[Tensor, float] = 0
    ) -> Tuple[Tensor, Tensor]:
        input_lb, input_ub = self.inputs.concretize()
        b_dim = input_lb.shape[0]
        dims = list(range(1, input_lb.dim()))
        dims.append(0)
        input_lb, input_ub = input_lb.permute(dims), input_ub.permute(
            dims
        )  # dim0, ... dimn, batch_dim,
        input_lb, input_ub = input_lb.view(-1, b_dim), input_ub.view(
            -1, b_dim
        )  # dim, batch_dim
        neg_x_l_coef, pos_x_l_coef = get_neg_pos_comp(
            self.x_l_coef[:, :, i] - self.x_u_coef[:, :, j]
        )
        neg_x_l_coef, pos_x_l_coef = neg_x_l_coef.view(-1, b_dim), pos_x_l_coef.view(
            -1, b_dim
        )
        delta = (
            self.x_l_bias[:, i]
            - self.x_u_bias[:, j]
            + (neg_x_l_coef * input_ub).sum(0)
            + (pos_x_l_coef * input_lb).sum(dim=0)
        )
        return delta, delta > threshold_min

    def verify(
        self,
        targets: Tensor,
        threshold_min: Union[Tensor, float] = 0,
        corr_only: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        n_class = self.x_l_bias.size()[1]
        device = self.x_l_bias.device
        dtype = self.x_l_coef.dtype
        verified = torch.zeros(targets.size(), dtype=torch.uint8).to(device)
        verified_corr = torch.zeros(targets.size(), dtype=torch.uint8).to(device)
        if n_class == 1:
            # assert len(targets) == 1
            verified_list = torch.cat(
                [
                    self.concretize()[1] < threshold_min,
                    self.concretize()[0] >= threshold_min,
                ],
                dim=1,
            )
            verified[:] = torch.any(verified_list, dim=1)
            verified_corr[:] = verified_list.gather(
                dim=1, index=targets.long().unsqueeze(dim=1)
            ).squeeze(1)
            threshold = (
                torch.cat(self.concretize(), 1)
                .gather(dim=1, index=(1 - targets).long().unsqueeze(dim=1))
                .squeeze(1)
            )
        else:
            threshold = np.inf * torch.ones(targets.size(), dtype=dtype).to(device)
            for i in range(n_class):
                if corr_only and i not in targets:
                    continue
                isg = torch.ones(targets.size(), dtype=torch.uint8).to(device)
                print(isg.shape)
                margin = np.inf * torch.ones(targets.size(), dtype=dtype).to(device)
                for j in range(n_class):
                    if i != j and isg.any():
                        margin_tmp, ok = self.is_greater(i, j, threshold_min)
                        margin = torch.min(margin, margin_tmp)
                        isg = isg & ok.byte()
                verified = verified | isg
                verified_corr = verified_corr | (targets.eq(i).byte() & isg)
                threshold = torch.where(targets.eq(i).byte(), margin, threshold)
        return verified, verified_corr, threshold

    def get_wc_logits(self, targets: Tensor, use_margins: bool = False) -> Tensor:
        n_class = self.shape[-1]
        device = self.x_l_coef.device
        dtype = self.x_l_coef.dtype

        if use_margins:

            def get_c_mat(n_class: int, target: Tensor) -> Tensor:
                return torch.eye(n_class, dtype=dtype)[target].unsqueeze(
                    dim=0
                ) - torch.eye(n_class, dtype=dtype)

            if n_class > 1:
                c = torch.stack([get_c_mat(n_class, x) for x in targets], dim=0)
                self = -(self.unsqueeze(dim=1) * c.to(device)).sum(
                    dim=2, reduce_dim=True
                )
        batch_size = targets.size()[0]
        lb, ub = self.concretize()
        if n_class == 1:
            wc_logits = torch.cat([ub, lb], dim=1)
            wc_logits = wc_logits.gather(dim=1, index=targets.long().unsqueeze(1))
        else:
            wc_logits = ub.clone()
            wc_logits[np.arange(batch_size), targets] = lb[
                np.arange(batch_size), targets
            ]
        return wc_logits

    def ce_loss(self, targets: Tensor) -> Tensor:
        wc_logits = self.get_wc_logits(targets)
        if wc_logits.size(1) == 1:
            return F.binary_cross_entropy_with_logits(
                wc_logits.squeeze(1), targets.float(), reduction="none"
            )
        else:
            return F.cross_entropy(wc_logits, targets.long(), reduction="none")

    def _get_multiplication_slopes_and_intercepts(
        self, mul_bounds: Tuple[Tensor, Tensor], input_bounds: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:

        input_lb, input_ub = input_bounds

        D = 1e-12 if input_lb.dtype == torch.float64 else 1e-7

        # Get the lower and upper bound of the multiplication
        (mult_lb_lb, mult_lb_ub, mult_ub_lb, mult_ub_ub) = self._get_mul_lbs_and_ubs(
            mul_bounds, input_bounds
        )

        # Get slopes and offsets
        # TODO look at effect of soundness correction here
        convex_lb_slope = (mult_ub_lb - mult_lb_lb) / (input_ub - input_lb + D)
        convex_lb_intercept = mult_lb_lb - input_lb * convex_lb_slope - D

        convex_ub_slope = (mult_ub_ub - mult_lb_ub) / (input_ub - input_lb + D)
        convex_ub_intercept = mult_lb_ub - input_lb * convex_ub_slope + D

        return (
            convex_lb_slope,
            convex_lb_intercept,
            convex_ub_slope,
            convex_ub_intercept,
        )

    def _get_mul_lbs_and_ubs(
        self, b1: Tuple[Tensor, Tensor], b2: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        input_lb_opts = [b2[0] * b1[0], b2[0] * b1[1]]
        input_ub_opts = [b2[1] * b1[0], b2[1] * b1[1]]
        mult_lb_lb = tensor_reduce(torch.minimum, input_lb_opts)
        mult_lb_ub = tensor_reduce(torch.maximum, input_lb_opts)
        mult_ub_lb = tensor_reduce(torch.minimum, input_ub_opts)
        mult_ub_ub = tensor_reduce(torch.maximum, input_ub_opts)
        return (mult_lb_lb, mult_lb_ub, mult_ub_lb, mult_ub_ub)
