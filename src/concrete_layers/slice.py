import torch
from torch import Tensor


# Slicing limited to 1-d slices wit positive steps
class Slice(torch.nn.Module):
    def __init__(self, dim: int, starts: int, ends: int, steps: int) -> None:
        super(Slice, self).__init__()
        self.starts = starts
        self.ends = ends
        self.dim = dim
        self.steps = steps

    def forward(self, x: Tensor) -> Tensor:
        axes = self.dim  # Expects input to have a batch-dimension
        starts = self.starts
        ends = self.ends
        steps = self.steps
        assert steps > 0
        if ends == -1:
            ends = x.shape[axes]

        index = torch.tensor(range(starts, ends, steps), device=x.device)
        selected = torch.index_select(x, axes, index)
        return selected
