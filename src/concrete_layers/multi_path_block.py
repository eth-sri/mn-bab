from typing import List, Optional

from torch import Tensor
from torch import nn as nn

from src.concrete_layers.binary_op import BinaryOp as concreteBinaryOp


class MultiPathBlock(nn.Module):
    def __init__(
        self, header: Optional[nn.Module], paths: List[nn.Sequential], merge: nn.Module
    ) -> None:
        super(MultiPathBlock, self).__init__()
        self.header = header
        self.paths = nn.ModuleList(paths)
        self.merge = merge

    def forward(self, x: Tensor) -> Tensor:

        if self.header:
            in_vals = self.header(x)
        else:
            in_vals = x

        if not isinstance(in_vals, list):
            in_vals = [in_vals for i in range(len(self.paths))]

        assert len(in_vals) == len(self.paths)

        out_vals: List[Tensor] = []
        for i, in_val in enumerate(in_vals):
            out_vals.append(self.paths[i](in_val))

        if issubclass(type(self.merge), concreteBinaryOp):
            out = self.merge(out_vals[0], out_vals[1])
        else:
            out = self.merge(out_vals)

        return out
