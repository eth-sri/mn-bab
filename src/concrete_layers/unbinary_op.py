import typing

import torch
from torch import Tensor


class UnbinaryOp(torch.nn.Module):
    def __init__(self, op: str, const_val: Tensor, apply_right: bool = False) -> None:
        super(UnbinaryOp, self).__init__()
        self.op = op
        self.register_buffer(
            "const_val",
            torch.as_tensor(const_val),
            persistent=False,
        )
        self.apply_right = apply_right

    @typing.no_type_check  # Mypy can't handle the buffer type
    def forward(self, x: Tensor) -> Tensor:
        assert type(self.const_val) == Tensor
        if self.const_val.device != x.device:
            self.to(x.device)
        if self.apply_right:
            left, right = self.const_val, x
        else:
            left, right = x, self.const_val

        if self.op == "add":
            return left + right
        elif self.op == "sub":
            return left - right
        elif self.op == "mul":
            return left * right
        elif self.op == "div":
            return left / right

    @typing.no_type_check
    def to(self, device: str) -> None:
        self.const_val = self.const_val.to(device)
