from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Mapping, Tuple

import torch
from torch import Tensor

from src.state.tags import LayerTag
from src.utilities.custom_typing import implement_properties_as_fields
from src.utilities.tensor_management import deep_copy_to, deep_copy_to_no_clone, move_to


class ReadonlyPrimaConstraints(ABC):
    """
    The PRIMA constraints are of the form:
     output_var_coefs @ layer_output + input_var_coefs @ layer_input + const_coefs @ 1 <= 0

    """

    @property
    @abstractmethod
    def prima_coefficients(self) -> Mapping[LayerTag, Tuple[Tensor, Tensor, Tensor]]:
        pass

    @property
    @abstractmethod
    def batch_size(self) -> int:
        pass

    @property
    @abstractmethod
    def device(self) -> torch.device:
        pass

    def deep_copy_to_no_clone(self, device: torch.device) -> ReadonlyPrimaConstraints:
        return PrimaConstraints(
            deep_copy_to_no_clone(self.prima_coefficients, device),
            self.batch_size,
            device,
        )

    def deep_copy_to(self, device: torch.device) -> PrimaConstraints:
        return PrimaConstraints(
            deep_copy_to(self.prima_coefficients, device), self.batch_size, device
        )

    @abstractmethod
    def set_prima_coefficients(
        self, prima_coefficients: Dict[LayerTag, Tuple[Tensor, Tensor, Tensor]]
    ) -> None:
        pass


@implement_properties_as_fields
class PrimaConstraints(ReadonlyPrimaConstraints):
    """
    The PRIMA constraints are of the form:
     output_var_coefs @ layer_output + input_var_coefs @ layer_input + const_coefs @ 1 <= 0

    """

    prima_coefficients: Dict[
        LayerTag, Tuple[Tensor, Tensor, Tensor]
    ]  # (output_var_coefs, input_var_coefs, const_coefs) (max 3 entries per row)
    batch_size: int
    device: torch.device

    def __init__(
        self,
        prima_coefficients: Dict[LayerTag, Tuple[Tensor, Tensor, Tensor]],
        batch_size: int,
        device: torch.device,
    ):
        self.prima_coefficients = prima_coefficients
        self.batch_size = batch_size
        self.device = device

    @classmethod
    def create_default(
        cls,
        batch_size: int,
        device: torch.device,
    ) -> PrimaConstraints:
        prima_coefficients: Dict[LayerTag, Tuple[Tensor, Tensor, Tensor]] = {}
        return cls(prima_coefficients, batch_size, device)

    def move_to(self, device: torch.device) -> None:
        if self.device is device:
            return
        self.prima_coefficients = move_to(self.prima_coefficients, device)
        self.device = device

    def set_prima_coefficients(
        self, prima_coefficients: Dict[LayerTag, Tuple[Tensor, Tensor, Tensor]]
    ) -> None:
        # assert prima_coefficients is self.prima_coefficients # TODO
        self.prima_coefficients = move_to(prima_coefficients, self.device)
