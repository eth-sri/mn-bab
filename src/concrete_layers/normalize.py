from typing import Sequence

import torch
from torch import Tensor


class Normalize(torch.nn.Module):
    means: Tensor
    stds: Tensor
    channel_dim: int

    def __init__(
        self, means: Sequence[float], stds: Sequence[float], channel_dim: int
    ) -> None:
        super(Normalize, self).__init__()
        target_shape = 4 * [1]
        target_shape[channel_dim] = len(means)
        self.register_buffer(
            "means",
            torch.as_tensor(means, dtype=torch.float).reshape(target_shape),
            persistent=False,
        )
        self.register_buffer(
            "stds",
            torch.as_tensor(stds, dtype=torch.float).reshape(target_shape),
            persistent=False,
        )
        self.channel_dim = channel_dim

    def forward(self, x: Tensor) -> Tensor:
        return (x - self.means) / self.stds
