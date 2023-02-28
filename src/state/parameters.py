from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

from src.state.layer_bounds import ReadonlyLayerBounds
from src.state.tags import LayerTag, ParameterTag, QueryTag
from src.utilities.custom_typing import implement_properties_as_fields
from src.utilities.tensor_management import (
    deep_copy,
    deep_copy_to,
    deep_copy_to_no_clone,
    move_to,
)


class ReadonlyParametersForQuery(ABC):
    @property
    @abstractmethod
    def parameters(self) -> Mapping[ParameterTag, Mapping[LayerTag, Tensor]]:
        pass

    @property
    @abstractmethod
    def batch_size(self) -> int:
        pass

    @property
    @abstractmethod
    def device(self) -> torch.device:
        pass

    def deep_copy(self) -> ParametersForQuery:
        return ParametersForQuery(
            deep_copy(self.parameters), self.batch_size, self.device
        )

    def deep_copy_to(self, device: torch.device) -> ParametersForQuery:
        return ParametersForQuery(
            deep_copy_to(self.parameters, device), self.batch_size, device
        )

    def deep_copy_to_no_clone(self, device: torch.device) -> ReadonlyParametersForQuery:
        return ParametersForQuery.create_readonly(
            deep_copy_to_no_clone(self.parameters, device),
            self.batch_size,
            device,
        )


@implement_properties_as_fields
class ParametersForQuery(ReadonlyParametersForQuery):
    parameters: Dict[ParameterTag, Dict[LayerTag, Tensor]]
    batch_size: int
    device: torch.device

    def __init__(
        self,
        parameters: Dict[ParameterTag, Dict[LayerTag, Tensor]],
        batch_size: int,
        device: torch.device,
    ):
        self.parameters = parameters
        self.batch_size = batch_size
        self.device = device

    @classmethod
    def create_readonly(
        cls,
        parameters: Mapping[ParameterTag, Mapping[LayerTag, Tensor]],
        batch_size: int,
        device: torch.device,
    ) -> ReadonlyParametersForQuery:
        assert isinstance(parameters, dict)
        return ParametersForQuery(parameters, batch_size, device)

    @classmethod
    def create_default(
        cls, batch_size: int, device: torch.device
    ) -> ParametersForQuery:
        parameters: Dict[ParameterTag, Dict[LayerTag, Tensor]] = {}
        return cls(parameters, batch_size, device)

    def move_to(self, device: torch.device) -> None:
        if self.device == device:
            return
        self.parameters = move_to(self.parameters, device)
        self.device = device

    def get_optimizable(
        self, only_lb: bool
    ) -> Tuple[
        List[Tensor], List[Tensor], List[Tensor], List[Tensor]
    ]:  # TODO: clean this up

        alpha_parameters_for_query: List[Tensor] = []
        beta_parameters_for_query: List[Tensor] = []
        prima_parameters_for_query: List[Tensor] = []
        alpha_relu_parameters_for_query: List[Tensor] = []

        for param_key, params_by_layer in self.parameters.items():
            if only_lb and "ub" in param_key:
                continue
            if "alpha_relu" in param_key:
                alpha_relu_parameters_for_query += list(params_by_layer.values())
            if "alpha" in param_key:
                alpha_parameters_for_query += list(params_by_layer.values())
            elif "beta" in param_key:
                beta_parameters_for_query += list(params_by_layer.values())
            elif "prima" in param_key:
                prima_parameters_for_query += list(params_by_layer.values())
            else:
                raise RuntimeError("Unknown optimizable parameter {}".format(param_key))

        return (
            alpha_parameters_for_query,
            beta_parameters_for_query,
            prima_parameters_for_query,
            alpha_relu_parameters_for_query,
        )

    def get_parameters(
        self,
        parameter_key: ParameterTag,
        layer_id: LayerTag,
        make_default_parameters: Union[
            Callable[[torch.device], Tensor],
            Tuple[int, ...],  # default is a zero tensor
        ],
    ) -> Tensor:
        parameters_per_layer = self.parameters.setdefault(parameter_key, {})
        if layer_id not in parameters_per_layer:
            if isinstance(make_default_parameters, tuple):
                parameter_shape = make_default_parameters
                default_parameters = torch.zeros(*parameter_shape, device=self.device)
            else:
                default_parameters = make_default_parameters(self.device)
            requested_parameters = default_parameters
            parameters_per_layer[layer_id] = requested_parameters
        else:
            requested_parameters = parameters_per_layer[layer_id]
        if not requested_parameters.requires_grad:
            requested_parameters.requires_grad_()
        # print(f"Here: {parameter_key}: {layer_id}")
        return requested_parameters

    def get_existing_parameters(
        self,
        parameter_key: ParameterTag,
        layer_id: LayerTag,
    ) -> Tensor:
        return self.parameters[parameter_key][layer_id]

    def modify_for_sharing(
        self,
    ) -> None:  # reduce query dimension to 1 so the parameters can be used without reduced parameter sharing
        for param_key, layer_parameters in self.parameters.items():
            if "alpha_relu" not in param_key:  # TODO: this is a bit hacky
                continue
            for layer_id, parameters in layer_parameters.items():
                if parameters.shape[1] != 1:
                    # just use the first parameter set
                    # select returns a view, intention of clone is to allow memory to be freed
                    layer_parameters[layer_id] = (
                        parameters.select(dim=1, index=0).unsqueeze(1).clone().detach()
                    )

    # (used for split score computations)
    def change_alphas_to_WK_slopes(self, layer_bounds: ReadonlyLayerBounds) -> None:
        from src.utilities.branching import (  # to avoid circular imports
            babsr_ratio_computation,
        )

        for param_key, layer_parameters in self.parameters.items():
            if "alpha" in param_key:
                for layer_id, parameters in layer_parameters.items():
                    current_layer_lower_bounds = layer_bounds.intermediate_bounds[
                        layer_id
                    ][0].unsqueeze(
                        1
                    )  # add query dimension
                    current_layer_upper_bounds = layer_bounds.intermediate_bounds[
                        layer_id
                    ][1].unsqueeze(
                        1
                    )  # add query dimension
                    ub_slope, __ = babsr_ratio_computation(
                        current_layer_lower_bounds, current_layer_upper_bounds
                    )
                    WK_slopes = ub_slope
                    self.parameters[param_key][layer_id] = WK_slopes

    def set_beta_parameters_to_zero(self) -> None:
        for param_key, layer_parameters in self.parameters.items():
            if "beta" in param_key:
                for layer_id, parameters in layer_parameters.items():
                    self.parameters[param_key][layer_id] = torch.zeros_like(parameters)

    def improve(
        self, new_parameters_for_query: ParametersForQuery, improvement_mask: Tensor
    ) -> None:
        # if not any(improvement_mask): return # (it suffices if this is done once in Parameters.improve for now)
        for (
            param_key,
            new_parameters_per_layer,
        ) in new_parameters_for_query.parameters.items():
            if param_key not in self.parameters:
                self.parameters[param_key] = {}
            for layer_id, new_layer_parameters in new_parameters_per_layer.items():
                if layer_id in self.parameters[param_key]:
                    improvement_mask_of_appropriate_shape = improvement_mask.view(
                        improvement_mask.shape[0],
                        *([1] * (len(new_layer_parameters.shape) - 1)),
                    )
                    self.parameters[param_key][layer_id] = torch.where(
                        improvement_mask_of_appropriate_shape,
                        new_layer_parameters,
                        self.parameters[param_key][layer_id],
                    ).detach()
                else:
                    self.parameters[param_key][layer_id] = deep_copy(
                        new_layer_parameters
                    )


class ReadonlyParameters(ABC):
    @property
    @abstractmethod
    def parameters_by_query(
        self,
    ) -> Mapping[QueryTag, ParametersForQuery]:
        pass

    @property
    @abstractmethod
    def batch_size(self) -> int:
        pass

    @property
    @abstractmethod
    def device(self) -> torch.device:
        pass

    @property
    @abstractmethod
    def use_params(
        self,
    ) -> bool:  # TODO: probably it would be better to move use_params into the backsubtitution config.
        pass

    def deep_copy(self) -> Parameters:
        return Parameters(
            {
                tag: params.deep_copy()
                for tag, params in self.parameters_by_query.items()
            },
            self.batch_size,
            self.device,
            self.use_params,
        )

    def deep_copy_to(self, device: torch.device) -> Parameters:
        return Parameters(
            {
                tag: params.deep_copy_to(device)
                for tag, params in self.parameters_by_query.items()
            },
            self.batch_size,
            device,
            self.use_params,
        )

    def deep_copy_to_no_clone(self, device: torch.device) -> ReadonlyParameters:
        return Parameters.create_readonly(
            {
                tag: params.deep_copy_to_no_clone(device)
                for tag, params in self.parameters_by_query.items()
            },
            self.batch_size,
            device,
            self.use_params,
        )

    def get_active_parameters_after_split(
        self,
        recompute_intermediate_bounds_after_branching: bool,
        intermediate_layer_bounds_to_be_kept_fixed: Sequence[int],
        device: torch.device,
    ) -> ReadonlyParameters:  # readonly because the resulting parameters alias the original ones
        active_parameters = (
            {
                query_id: parameters
                for query_id, parameters in self.parameters_by_query.items()
                if query_id not in intermediate_layer_bounds_to_be_kept_fixed
            }
            if not recompute_intermediate_bounds_after_branching
            else self.parameters_by_query
        )
        return Parameters.create_readonly(
            active_parameters, self.batch_size, device, self.use_params
        )


@implement_properties_as_fields
class Parameters(ReadonlyParameters):
    parameters_by_query: Dict[QueryTag, ParametersForQuery]
    batch_size: int
    device: torch.device
    use_params: bool  # only initialize parameters if this is True => false for DP pass # TODO: probably it's better to move use_params into the backsubstitution config

    def __init__(
        self,
        parameters_by_query: Dict[QueryTag, ParametersForQuery],
        batch_size: int,
        device: torch.device,
        use_params: bool,
    ):
        self.parameters_by_query = parameters_by_query
        self.batch_size = batch_size
        self.device = device
        self.use_params = use_params

    @classmethod
    def create_readonly(
        cls,
        parameters_by_query: Mapping[QueryTag, ReadonlyParametersForQuery],
        batch_size: int,
        device: torch.device,
        use_params: bool,
    ) -> ReadonlyParameters:
        assert isinstance(parameters_by_query, dict)
        return Parameters(parameters_by_query, batch_size, device, use_params)

    @classmethod
    def create_default(
        cls, batch_size: int, device: torch.device, use_params: bool
    ) -> Parameters:
        parameters_by_query: Dict[QueryTag, ParametersForQuery] = {}
        return cls(parameters_by_query, batch_size, device, use_params)

    def move_to(self, device: torch.device) -> None:
        if self.device == device:
            return
        for parameters_for_query in self.parameters_by_query.values():
            parameters_for_query.move_to(device)
        self.device = device

    def get_parameters_for_query(self, query_id: QueryTag) -> ParametersForQuery:
        if query_id in self.parameters_by_query:
            return self.parameters_by_query[query_id]
        result = ParametersForQuery.create_default(self.batch_size, self.device)
        self.parameters_by_query[query_id] = result
        return result

    def get_parameters(
        self,
        query_id: QueryTag,
        parameter_key: ParameterTag,
        layer_id: LayerTag,
        make_default_parameters: Union[
            Callable[[torch.device], Tensor],
            Tuple[int, ...],  # default is a zero tensor
        ],
    ) -> Tensor:
        parameters_for_query = self.get_parameters_for_query(query_id)
        return parameters_for_query.get_parameters(
            parameter_key=parameter_key,
            layer_id=layer_id,
            make_default_parameters=make_default_parameters,
        )

    def get_existing_parameters(
        self, query_id: QueryTag, parameter_key: ParameterTag, layer_id: LayerTag
    ) -> Tensor:
        parameters_for_query = self.get_parameters_for_query(query_id)
        return parameters_for_query.get_existing_parameters(
            parameter_key=parameter_key,
            layer_id=layer_id,
        )

    def modify_for_sharing(self) -> None:
        for query_id, parameters_for_query in self.parameters_by_query.items():
            parameters_for_query.modify_for_sharing()

    def change_alphas_to_WK_slopes(
        self, query_id: QueryTag, layer_bounds: ReadonlyLayerBounds
    ) -> None:
        parameters_for_query = self.get_parameters_for_query(query_id)
        parameters_for_query.change_alphas_to_WK_slopes(layer_bounds)

    def set_beta_parameters_to_zero(self, query_id: QueryTag) -> None:
        parameters_for_query = self.get_parameters_for_query(query_id)
        parameters_for_query.set_beta_parameters_to_zero()

    def get_optimizable(
        self, selected_query_id: Optional[QueryTag], only_lb: bool
    ) -> Tuple[
        List[Tensor], List[Tensor], List[Tensor], List[Tensor]
    ]:  # TODO: clean this up

        all_alpha_parameters: List[Tensor] = []
        all_beta_parameters: List[Tensor] = []
        all_prima_parameters: List[Tensor] = []
        all_alpha_relu_parameters: List[Tensor] = []

        for (
            layer_id,
            parameters_for_query,
        ) in self.parameters_by_query.items():
            if selected_query_id is not None and layer_id != selected_query_id:
                continue
            (
                alpha_parameters_for_query,
                beta_parameters_for_query,
                prima_parameters_for_query,
                alpha_relu_parameters_for_query,
            ) = parameters_for_query.get_optimizable(only_lb)

            all_alpha_parameters += alpha_parameters_for_query
            all_beta_parameters += beta_parameters_for_query
            all_prima_parameters += prima_parameters_for_query
            all_alpha_relu_parameters += alpha_relu_parameters_for_query

        return (
            all_alpha_parameters,
            all_beta_parameters,
            all_prima_parameters,
            all_alpha_relu_parameters,
        )

    def improve(self, new_parameters: Parameters, improvement_mask: Tensor) -> None:
        if not any(improvement_mask):
            return
        for (
            query_id,
            new_parameters_for_query,
        ) in new_parameters.parameters_by_query.items():
            if query_id not in self.parameters_by_query:
                self.parameters_by_query[query_id] = ParametersForQuery.create_default(
                    self.batch_size, self.device
                )
            self.parameters_by_query[query_id].improve(
                new_parameters_for_query, improvement_mask
            )
