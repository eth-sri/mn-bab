from collections import OrderedDict
from typing import Any

import torch


def deep_copy(obj: Any) -> Any:
    if obj is None:
        return obj
    if torch.is_tensor(obj):
        assert obj.is_leaf
        return obj.clone().detach()
    if isinstance(obj, OrderedDict):
        return OrderedDict((k, deep_copy(v)) for k, v in obj.items())
    if isinstance(obj, dict):
        return {k: deep_copy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [deep_copy(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(deep_copy(v) for v in obj)
    raise TypeError("Invalid type '" + str(type(obj)) + "' for deep_copy")


def deep_copy_to(obj: Any, device: torch.device) -> Any:
    """
    Clones built-in Python data structures containing leaf Tensors
    to the specified device. Result does not alias argument.
    Use this if object is not owned or should be copied.
    """
    if obj is None:
        return obj
    if torch.is_tensor(obj):
        assert obj.is_leaf
        if obj.device == device:
            return obj.clone().detach()
        return obj.to(device).detach()
    if isinstance(obj, OrderedDict):
        return OrderedDict((k, deep_copy_to(v, device)) for k, v in obj.items())
    if isinstance(obj, dict):
        return {k: deep_copy_to(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [deep_copy_to(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(deep_copy_to(v, device) for v in obj)
    raise TypeError("Invalid type '" + str(type(obj)) + "' for deep_copy_to")


def move_to(obj: Any, device: torch.device) -> Any:
    """
    Moves built-in Python data structures containing leaf Tensors
    to the specified device. obj is updated in place if possible.
    Use this for owned objects.
    """
    if obj is None:
        return
    if torch.is_tensor(obj):  # tensor, cannot update in place
        assert obj.is_leaf
        return obj.to(device).detach()
    if isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = move_to(v, device)
        return obj
    if isinstance(obj, list):
        for i in range(len(obj)):
            obj[i] = move_to(obj[i], device)
        return obj
    if isinstance(obj, tuple):  # tuple, cannot update in place
        return tuple(move_to(v, device) for v in obj)
    raise TypeError("Invalid type '" + str(type(obj)) + "' for move_to")


def deep_copy_to_no_clone(obj: Any, device: torch.device) -> Any:
    """
    Copies Python data structures containing leaf tensors, but does
    not make guarantees about aliasing. Use with caution when Tensor
    aliasing is unimportant and Tensors just need to go to the
    respective device. Probably in almost all cases you want to use
    deep_copy_to or move_to instead.
    """
    if obj is None:
        return obj
    if torch.is_tensor(obj):
        assert obj.is_leaf
        return obj.to(device).detach()
    if isinstance(obj, OrderedDict):
        return OrderedDict(
            (k, deep_copy_to_no_clone(v, device)) for k, v in obj.items()
        )
    if isinstance(obj, dict):
        return {k: deep_copy_to_no_clone(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [deep_copy_to_no_clone(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(deep_copy_to_no_clone(v, device) for v in obj)
    raise TypeError("Invalid type '" + str(type(obj)) + "' for deep_copy_to_no_clone")
