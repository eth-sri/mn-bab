from typing import Type

from torch import nn as nn

from src.abstract_layers.abstract_basic_block import BasicBlock
from src.abstract_layers.abstract_bn2d import BatchNorm2d
from src.abstract_layers.abstract_conv2d import Conv2d
from src.abstract_layers.abstract_flatten import Flatten
from src.abstract_layers.abstract_identity import Identity
from src.abstract_layers.abstract_linear import Linear
from src.abstract_layers.abstract_normalization import Normalization
from src.abstract_layers.abstract_relu import ReLU
from src.abstract_layers.abstract_sequential import Sequential
from src.concrete_layers import basic_block as concrete_basic_block
from src.concrete_layers import normalize as concrete_normalize


class AbstractModuleMapper:
    @staticmethod
    def map_to_abstract_type(concrete_type: Type) -> Type:
        if concrete_type == nn.BatchNorm2d:
            return BatchNorm2d
        elif concrete_type == nn.Conv2d:
            return Conv2d
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
        elif concrete_type == concrete_basic_block.BasicBlock:
            return BasicBlock
        elif concrete_type == nn.Sequential:
            return Sequential
        else:
            raise NotImplementedError(f"Unsupported layer type: {concrete_type}")
