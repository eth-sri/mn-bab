from typing import Type

from torch import nn as nn

from src.abstract_layers.abstract_avg_pool2d import AvgPool2d
from src.abstract_layers.abstract_bn2d import BatchNorm2d
from src.abstract_layers.abstract_concat import Concat
from src.abstract_layers.abstract_conv2d import Conv2d
from src.abstract_layers.abstract_convtranspose2d import ConvTranspose2d
from src.abstract_layers.abstract_flatten import Flatten
from src.abstract_layers.abstract_identity import Identity
from src.abstract_layers.abstract_linear import Linear
from src.abstract_layers.abstract_max_pool2d import MaxPool2d
from src.abstract_layers.abstract_mulit_path_block import MultiPathBlock
from src.abstract_layers.abstract_normalization import Normalization
from src.abstract_layers.abstract_pad import Pad
from src.abstract_layers.abstract_permute import Permute
from src.abstract_layers.abstract_relu import ReLU
from src.abstract_layers.abstract_reshape import Reshape
from src.abstract_layers.abstract_residual_block import ResidualBlock
from src.abstract_layers.abstract_sequential import Sequential
from src.abstract_layers.abstract_sigmoid import Sigmoid
from src.abstract_layers.abstract_slice import Slice
from src.abstract_layers.abstract_split_block import SplitBlock
from src.abstract_layers.abstract_tanh import Tanh
from src.abstract_layers.abstract_unbinary_op import UnbinaryOp
from src.concrete_layers import basic_block as concrete_basic_block
from src.concrete_layers import concat as concrete_concat
from src.concrete_layers import multi_path_block as concrete_multi_path_block
from src.concrete_layers import normalize as concrete_normalize
from src.concrete_layers import pad as concrete_pad
from src.concrete_layers import permute as concrete_permute
from src.concrete_layers import reshape as concrete_reshape
from src.concrete_layers import residual_block as concrete_residual_block
from src.concrete_layers import slice as concrete_slice
from src.concrete_layers import split_block as concrete_split_block
from src.concrete_layers import unbinary_op as concrete_unbinary_op


class AbstractModuleMapper:
    @staticmethod  # noqa: C901
    def map_to_abstract_type(concrete_type: Type) -> Type:
        if concrete_type == nn.BatchNorm2d:
            return BatchNorm2d
        elif concrete_type == concrete_concat.Concat:
            return Concat
        elif concrete_type == nn.Conv2d:
            return Conv2d
        elif concrete_type == nn.ConvTranspose2d:
            return ConvTranspose2d
        elif concrete_type == nn.Flatten:
            return Flatten
        elif concrete_type == nn.Identity:
            return Identity
        elif concrete_type == nn.Linear:
            return Linear
        elif concrete_type == concrete_normalize.Normalize:
            return Normalization
        elif concrete_type == nn.ReLU:
            return ReLU
        elif concrete_type == nn.Sigmoid:
            return Sigmoid
        elif concrete_type == concrete_slice.Slice:
            return Slice
        elif concrete_type == nn.Tanh:
            return Tanh
        elif concrete_type == concrete_unbinary_op.UnbinaryOp:
            return UnbinaryOp
        elif concrete_type == concrete_basic_block.BasicBlock:
            return ResidualBlock
        elif concrete_type == concrete_residual_block.ResidualBlock:
            return ResidualBlock
        elif concrete_type == nn.Sequential:
            return Sequential
        elif concrete_type == concrete_permute.Permute:
            return Permute
        elif concrete_type == concrete_split_block.SplitBlock:
            return SplitBlock
        elif concrete_type == nn.AvgPool2d:
            return AvgPool2d
        elif concrete_type == concrete_multi_path_block.MultiPathBlock:
            return MultiPathBlock
        elif concrete_type == nn.MaxPool2d:
            return MaxPool2d
        elif concrete_type == concrete_pad.Pad:
            return Pad
        elif concrete_type == concrete_reshape.Reshape:
            return Reshape
        else:
            raise NotImplementedError(f"Unsupported layer type: {concrete_type}")
