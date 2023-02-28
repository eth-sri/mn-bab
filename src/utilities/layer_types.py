from src.abstract_layers.abstract_conv2d import Conv2d
from src.abstract_layers.abstract_linear import Linear
from src.abstract_layers.abstract_module import AbstractModule
from src.utilities.config import LayerType


def is_layer_of_type(prev_layer: AbstractModule, layer_type: LayerType) -> bool:
    if layer_type == LayerType.fully_connected:
        return isinstance(prev_layer, Linear)
    if layer_type == LayerType.conv2d:
        return isinstance(prev_layer, Conv2d)
    raise Exception("unsupported layer type '" + str(layer_type.value) + "'")
