from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import torch
from torch import Tensor

if TYPE_CHECKING:
    from src.abstract_layers.abstract_bn2d import BatchNorm2d


def clamp_image(
    x: Tensor, eps: Union[Tensor, float], clamp_min: float = 0, clamp_max: float = 1
) -> Tuple[Tensor, Tensor]:
    min_x = torch.clamp(x - eps, min=clamp_min)
    max_x = torch.clamp(x + eps, max=clamp_max)
    x_center = 0.5 * (max_x + min_x)
    x_beta = 0.5 * (max_x - min_x)
    return x_center, x_beta


def head_from_bounds(min_x: Tensor, max_x: Tensor) -> Tuple[Tensor, Tensor]:
    x_center = 0.5 * (max_x + min_x)
    x_betas = 0.5 * (max_x - min_x)
    return x_center, x_betas


class AbstractElement:
    def __init__(self) -> None:
        pass

    def __neg__(self) -> AbstractElement:
        raise NotImplementedError

    def __sub__(
        self, other: Union[Tensor, float, int, "AbstractElement"]
    ) -> "AbstractElement":
        raise NotImplementedError

    def __add__(
        self, other: Union[Tensor, float, int, "AbstractElement"]
    ) -> "AbstractElement":
        raise NotImplementedError

    @property
    def shape(self) -> torch.Size:
        raise NotImplementedError

    @property
    def dtype(self) -> torch.dtype:
        raise NotImplementedError

    @property
    def device(self) -> torch.device:
        raise NotImplementedError

    @staticmethod
    def cat(x: List["AbstractElement"], dim: int) -> "AbstractElement":
        raise NotImplementedError

    def max_center(self) -> Tensor:
        raise NotImplementedError

    def conv2d(
        self,
        weight: Tensor,
        bias: Optional[Tensor],
        stride: int,
        padding: int,
        dilation: int,
        groups: int,
    ) -> "AbstractElement":
        raise NotImplementedError

    def convtranspose2d(
        self,
        weight: Tensor,
        bias: Optional[Tensor],
        stride: int,
        padding: int,
        out_padding: int,
        groups: int,
        dilation: int,
    ) -> "AbstractElement":
        raise NotImplementedError

    def avg_pool2d(
        self, kernel_size: int, stride: int, padding: int
    ) -> "AbstractElement":
        raise NotImplementedError

    def batch_norm(self, bn: BatchNorm2d) -> "AbstractElement":
        raise NotImplementedError

    def einsum(self, defining_str: str, other: Tensor) -> AbstractElement:
        raise NotImplementedError

    def flatten(self) -> "AbstractElement":
        raise NotImplementedError

    def max_pool2d(
        self,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int],
        padding: Tuple[int, int],
    ) -> "AbstractElement":
        raise NotImplementedError

    def pad(
        self, kernel_size: Tuple[int, ...], mode: str, value: float
    ) -> "AbstractElement":
        raise NotImplementedError

    def upsample(
        self, size: int, mode: str, align_corners: bool, consolidate_errors: bool
    ) -> "AbstractElement":
        raise NotImplementedError

    def linear(self, weight: Tensor, bias: Tensor) -> "AbstractElement":
        raise NotImplementedError

    def size(self) -> Union[Tuple[int, ...], int]:
        raise NotImplementedError

    def sum(self, dim: int, reduce_dim: bool) -> "AbstractElement":
        raise NotImplementedError

    def view(self, shape_tuple: Tuple[int, ...]) -> "AbstractElement":
        raise NotImplementedError

    def multiply_interval(self, interval: Tuple[Tensor, Tensor]) -> "AbstractElement":
        raise NotImplementedError

    def normalize(self, mean: Tensor, sigma: Tensor) -> "AbstractElement":
        raise NotImplementedError

    def clone(self) -> "AbstractElement":
        raise NotImplementedError

    def relu(
        self,
        deepz_lambda: Optional[Tensor] = None,
        bounds: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple["AbstractElement", Optional[Tensor]]:
        raise NotImplementedError

    def sigmoid(
        self,
        step_size: float,
        max_x: float,
        deepz_lambda: Optional[Tensor] = None,
        bounds: Optional[Tuple[Tensor, Tensor]] = None,
        tangent_points: Optional[Tensor] = None,
    ) -> Tuple["AbstractElement", Optional[Tensor]]:
        raise NotImplementedError

    def slice(
        self,
        dim: int,
        starts: int,
        ends: int,
        steps: int,
    ) -> "AbstractElement":
        raise NotImplementedError

    def split(
        self, split_size_or_sections: Tuple[int, ...], dim: int
    ) -> Tuple[AbstractElement, ...]:
        raise NotImplementedError

    def concretize(self) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError

    def evaluate_queries(
        self, query_matrix: Tensor, query_threshold: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, AbstractElement]:
        # Verify query_matrix * x > query_threshold
        # Assumes query is identical for all batch elements

        if query_threshold is None:
            query_threshold = torch.zeros_like(query_matrix[:, 0])

        abs_query = self.einsum(
            "bs, bqs -> bq", query_matrix.to(self.device).to(self.dtype)
        ) - query_threshold.to(self.device).to(self.dtype)
        query_lb, query_ub = abs_query.concretize()
        verified = query_lb > 0  # .view(-1)
        falsified = query_ub < 0  # .view(-1)

        assert not (
            verified.__and__(falsified)
        ).any(), "Should never verify and falsify a property"

        return verified, falsified, query_lb, query_ub, abs_query

    def may_contain_point(self, x: Tensor, D: float = 1e-7) -> bool:
        lb, ub = self.concretize()
        may_contain = (lb <= x + D).__and__(x - D <= ub).all()
        if not may_contain:
            print(f"Max violation lb: {(lb-x).max()}; Max violation ub: {(x-ub).max()}")
        return may_contain  # type: ignore # this is a bool


def get_neg_pos_comp(x: Tensor) -> Tuple[Tensor, Tensor]:
    neg_comp = torch.where(x < 0, x, torch.zeros_like(x))
    pos_comp = torch.where(x >= 0, x, torch.zeros_like(x))
    return neg_comp, pos_comp
