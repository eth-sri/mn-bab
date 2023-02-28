import torch
from torch import Tensor


# Concrete Implementation of BinaryOp used so that we can have an abstract version that spawns multiple shapes
class BinaryOp(torch.nn.Module):
    def __init__(self, op: str) -> None:
        super(BinaryOp, self).__init__()
        self.op = op

    def forward(self, x: Tensor, y: Tensor) -> Tensor:

        if self.op == "add":
            return x + y
        elif self.op == "sub":
            return x - y
        else:
            assert False, f"Unknown operator {self.op}"
