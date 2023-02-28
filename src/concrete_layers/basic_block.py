from typing import List, Tuple

import torch
from torch import nn as nn

from src.concrete_layers.residual_block import ResidualBlock


class BasicBlock(ResidualBlock, nn.Module):
    expansion = 1

    in_planes: int
    planes: int
    stride: int
    bn: bool
    kernel: int

    out_dim: int

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        bn: bool = True,
        kernel: int = 3,
        in_dim: int = -1,
    ) -> None:
        self.in_planes = in_planes
        self.planes = planes
        self.stride = stride
        self.bn = bn
        self.kernel = kernel

        kernel_size = kernel
        assert kernel_size in [1, 2, 3], "kernel not supported!"
        p_1 = 1 if kernel_size > 1 else 0
        p_2 = 1 if kernel_size > 2 else 0

        layers_b: List[nn.Module] = []
        layers_b.append(
            nn.Conv2d(
                in_planes,
                planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=p_1,
                bias=(not bn),
            )
        )
        _, _, in_dim = self._getShapeConv(
            (in_planes, in_dim, in_dim),
            (self.in_planes, kernel_size, kernel_size),
            stride=stride,
            padding=p_1,
        )

        if bn:
            layers_b.append(nn.BatchNorm2d(planes))
        layers_b.append(nn.ReLU())
        layers_b.append(
            nn.Conv2d(
                planes,
                self.expansion * planes,
                kernel_size=kernel_size,
                stride=1,
                padding=p_2,
                bias=(not bn),
            )
        )
        _, _, in_dim = self._getShapeConv(
            (planes, in_dim, in_dim),
            (self.in_planes, kernel_size, kernel_size),
            stride=1,
            padding=p_2,
        )
        if bn:
            layers_b.append(nn.BatchNorm2d(self.expansion * planes))
        path_b = nn.Sequential(*layers_b)

        layers_a: List[nn.Module] = [torch.nn.Identity()]
        if stride != 1 or in_planes != self.expansion * planes:
            layers_a.append(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=(not bn),
                )
            )
            if bn:
                layers_a.append(nn.BatchNorm2d(self.expansion * planes))
        path_a = nn.Sequential(*layers_a)
        self.out_dim = in_dim
        super(BasicBlock, self).__init__(path_a, path_b)

    def _getShapeConv(
        self,
        in_shape: Tuple[int, int, int],
        conv_shape: Tuple[int, ...],
        stride: int = 1,
        padding: int = 0,
    ) -> Tuple[int, int, int]:
        inChan, inH, inW = in_shape
        outChan, kH, kW = conv_shape[:3]

        outH = 1 + int((2 * padding + inH - kH) / stride)
        outW = 1 + int((2 * padding + inW - kW) / stride)
        return (outChan, outH, outW)
