from typing import Any, Tuple

import torch
from torch import Tensor
from torch.autograd import Function


class LeakyGradientMinimumFunction(Function):
    @staticmethod
    def forward(ctx: Any, input: Tensor, other: Tensor) -> Tensor:
        return torch.minimum(input, other)

    @staticmethod
    def backward(ctx: Any, grad_outputs: Tensor) -> Tuple[Tensor, Tensor]:
        return grad_outputs, grad_outputs
