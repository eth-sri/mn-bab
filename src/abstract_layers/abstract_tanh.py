from __future__ import annotations

import os
from typing import Any, Callable, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from src.abstract_layers.abstract_module import AbstractModule
from src.abstract_layers.abstract_sig_base import SigBase
from src.mn_bab_shape import MN_BaB_Shape
from src.state.tags import ParameterTag, layer_tag
from src.utilities.bilinear_interpolator import BilinearInterpol
from src.utilities.config import BacksubstitutionConfig


def tanh(x: Tensor) -> Tensor:
    return torch.tanh(x)


def d_tanh(x: Tensor) -> Tensor:
    return 1 - torch.tanh(x) * torch.tanh(x)


FILE_DIR = os.path.realpath(os.path.dirname(__file__))


class Tanh(SigBase, AbstractModule):

    sp_interpolator: Optional[BilinearInterpol] = None
    intersection_points: Optional[Tensor] = None
    tangent_points: Optional[Tensor] = None
    step_size: Optional[float] = None
    max_x: Optional[float] = None

    def __init__(self, dim: Tuple[int, ...]) -> None:
        super(Tanh, self).__init__(dim, tanh, d_tanh)
        if Tanh.intersection_points is None:
            (
                Tanh.intersection_points,
                Tanh.tangent_points,
                Tanh.step_size,
                Tanh.max_x,
            ) = SigBase._compute_bound_to_tangent_point(tanh, d_tanh)
        if Tanh.sp_interpolator is None:
            Tanh.sp_interpolator = BilinearInterpol.load_from_path(
                os.path.realpath(
                    os.path.join(FILE_DIR, "../../data/tanh_bil_interpol.pkl")
                )
            )
        self.output_dim = dim
        self.dependence_set_block = False

    @classmethod
    def from_concrete_module(  # type: ignore[override] # (checked at runtime)
        cls, module: nn.Tanh, input_dim: Tuple[int, ...], **kwargs: Any
    ) -> Tanh:
        assert isinstance(module, nn.Tanh)
        return cls(input_dim)

    def backsubstitute(
        self,
        config: BacksubstitutionConfig,
        abstract_shape: MN_BaB_Shape,
        intermediate_bounds_callback: Optional[
            Callable[[Tensor], Tuple[Tensor, Tensor]]
        ] = None,
        prev_layer: Optional[AbstractModule] = None,
    ) -> MN_BaB_Shape:

        # TODO solve better
        if (
            self.tangent_points is not None
            and self.tangent_points.device != abstract_shape.device
        ):
            self.tangent_points = self.tangent_points.to(device=abstract_shape.device)
        if (
            self.tangent_points is not None
            and self.tangent_points.dtype != abstract_shape.lb.bias.dtype
        ):
            self.tangent_points = self.tangent_points.to(
                dtype=abstract_shape.lb.bias.dtype
            )
        return super(Tanh, self)._backsubstitute(
            abstract_shape,
            self.tangent_points,
            self.step_size,
            self.max_x,
            intermediate_bounds_callback,
        )

    def get_approximation_slopes_and_intercepts(
        self,
        bounds: Tuple[Tensor, Tensor],
        abstract_shape: Optional[MN_BaB_Shape] = None,
        parameter_key: Optional[ParameterTag] = None,
        split_constraints: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        if self.tangent_points is None or self.step_size is None or self.max_x is None:
            raise RuntimeError(
                "Cannot compute Sig/Tanh bounds without pre-computed values"
            )
        if (
            abstract_shape is not None
            and self.tangent_points.device != abstract_shape.device
        ):
            self.tangent_points = self.tangent_points.to(device=abstract_shape.device)
        return super(Tanh, self)._get_approximation_slopes_and_intercepts_for_act(
            bounds,
            self.tangent_points,
            self.step_size,
            self.max_x,
            tanh,
            d_tanh,
            abstract_shape,
            parameter_key,
            layer_tag(self),
            split_constraints,
        )

    def forward(self, x: Tensor) -> Tensor:
        return torch.tanh(x)

    @classmethod
    def get_split_points(cls, lb: Tensor, ub: Tensor) -> Tensor:
        assert cls.sp_interpolator, "Split point interpolator for Tanh not set"
        return cls.sp_interpolator.get_value(lb, ub)
