from __future__ import annotations

from typing import Any, Tuple

from torch import Tensor

from src.abstract_layers.abstract_container_module import AbstractContainerModule
from src.abstract_layers.abstract_sequential import Sequential
from src.concrete_layers import basic_block as concrete_basic_block
from src.concrete_layers.basic_block import BasicBlock as concreteBasicBlock


class BasicBlock(
    concreteBasicBlock, AbstractContainerModule
):  # TODO: should this inherit from abstract ResidualBlock?
    path_a: Sequential  # type: ignore[assignment] # hack
    path_b: Sequential  # type: ignore[assignment]
    output_dim: Tuple[int, ...]
    bias: Tensor

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int,
        bn: bool,
        kernel: int,
        input_dim: Tuple[int, ...],
        **kwargs: Any,
    ) -> None:
        super(BasicBlock, self).__init__(in_planes, planes, stride, bn, kernel)
        self.path_a = Sequential.from_concrete_module(self.path_a, input_dim, **kwargs)  # type: ignore[arg-type] # hack
        self.path_b = Sequential.from_concrete_module(self.path_b, input_dim, **kwargs)  # type: ignore[arg-type] # hack
        self.output_dim = self.path_b.layers[-1].output_dim
        self.bias = self.get_babsr_bias()

    @classmethod
    def from_concrete_module(  # type: ignore[override] # (checked at runtime)
        cls,
        module: concrete_basic_block.BasicBlock,
        input_dim: Tuple[int, ...],
        **kwargs: Any,
    ) -> BasicBlock:
        assert isinstance(module, concrete_basic_block.BasicBlock)
        abstract_layer = cls(
            module.in_planes,
            module.planes,
            module.stride,
            module.bn,
            module.kernel,
            input_dim,
            **kwargs,
        )
        abstract_layer.path_a = Sequential.from_concrete_module(
            module.path_a, input_dim, **kwargs
        )
        abstract_layer.path_b = Sequential.from_concrete_module(
            module.path_b, input_dim, **kwargs
        )
        abstract_layer.bias = abstract_layer.get_babsr_bias()
        return abstract_layer

    def get_babsr_bias(self) -> Tensor:
        raise NotImplementedError  # TODO: inherit from abstract ResidualBlock?
