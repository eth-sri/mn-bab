from __future__ import annotations

import os
from typing import Any, Callable, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from src.abstract_domains.ai_util import AbstractElement
from src.abstract_layers.abstract_module import AbstractModule
from src.abstract_layers.abstract_sig_base import SigBase
from src.mn_bab_shape import MN_BaB_Shape
from src.state.tags import ParameterTag, layer_tag
from src.utilities.bilinear_interpolator import BilinearInterpol
from src.utilities.config import BacksubstitutionConfig


def sig(x: Tensor) -> Tensor:
    return torch.sigmoid(x)


def d_sig(x: Tensor) -> Tensor:
    sig = torch.sigmoid(x)
    return sig * (1 - sig)


FILE_DIR = os.path.realpath(os.path.dirname(__file__))


class Sigmoid(SigBase, AbstractModule):

    sp_interpolator: Optional[BilinearInterpol] = None
    intersection_points: Optional[Tensor] = None
    tangent_points: Optional[Tensor] = None
    step_size: Optional[float] = None
    max_x: Optional[float] = None

    def __init__(self, dim: Tuple[int, ...]) -> None:
        super(Sigmoid, self).__init__(dim, sig, d_sig)
        if Sigmoid.intersection_points is None:
            (
                Sigmoid.intersection_points,
                Sigmoid.tangent_points,
                Sigmoid.step_size,
                Sigmoid.max_x,
            ) = SigBase._compute_bound_to_tangent_point(sig, d_sig)
        if Sigmoid.sp_interpolator is None:
            Sigmoid.sp_interpolator = BilinearInterpol.load_from_path(
                os.path.realpath(
                    os.path.join(FILE_DIR, "../../data/sig_bil_interpol.pkl")
                )
            )
        self.output_dim = dim
        self.dependence_set_block = False

    @classmethod
    def from_concrete_module(  # type: ignore[override] # (checked at runtime)
        cls, module: nn.Sigmoid, input_dim: Tuple[int, ...], **kwargs: Any
    ) -> Sigmoid:
        assert isinstance(module, nn.Sigmoid)
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
        assert self.tangent_points is not None, "Tangent points not set"
        if self.tangent_points.device != abstract_shape.device:
            self.tangent_points = self.tangent_points.to(device=abstract_shape.device)
        if self.tangent_points.dtype != abstract_shape.lb.bias.dtype:
            self.tangent_points = self.tangent_points.to(
                dtype=abstract_shape.lb.bias.dtype
            )
        return super(Sigmoid, self)._backsubstitute(
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
        elif self.tangent_points.device != bounds[0].device:
            self.tangent_points = self.tangent_points.to(device=bounds[0].device)
        return super(Sigmoid, self)._get_approximation_slopes_and_intercepts_for_act(
            bounds,
            self.tangent_points,
            self.step_size,
            self.max_x,
            sig,
            d_sig,
            abstract_shape,
            parameter_key,
            layer_tag(self),
            split_constraints,
        )

    @classmethod
    def get_split_points(cls, lb: Tensor, ub: Tensor) -> Tensor:
        assert cls.sp_interpolator, "Split point interpolator for Sigmoid not set"
        return cls.sp_interpolator.get_value(lb, ub)

    def propagate_abstract_element(
        self,
        abs_input: AbstractElement,
        use_existing_bounds: Optional[bool] = None,
        activation_layer_only: bool = False,
        set_input: bool = True,
        set_output: bool = True,
    ) -> AbstractElement:
        assert self.step_size is not None
        assert self.max_x is not None
        return abs_input.sigmoid(
            tangent_points=self.tangent_points,
            step_size=self.step_size,
            max_x=self.max_x,
        )[0]
