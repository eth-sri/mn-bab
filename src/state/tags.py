from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import TYPE_CHECKING, NewType, Tuple

if TYPE_CHECKING:
    from src.abstract_layers.abstract_module import AbstractModule

# def query_tag_for_intermediate_bounds(...) # TODO
# def query_tag_for_final_result(...)        # TODO


ParameterTag = NewType(
    "ParameterTag", str
)  # type of parameter (e.g., {alpha, beta, prima}_{lb,ub})

key_alpha_lb = ParameterTag("alpha_lb")
key_alpha_ub = ParameterTag("alpha_ub")


def key_alpha(compute_upper_bound: bool) -> ParameterTag:
    return key_alpha_ub if compute_upper_bound else key_alpha_lb


def key_plus_lb(key: ParameterTag) -> ParameterTag:
    assert key == key_alpha_lb or key == key_alpha_ub
    return ParameterTag(key + "_lb")


def key_plus_ub(key: ParameterTag) -> ParameterTag:
    assert key == key_alpha_lb or key == key_alpha_ub
    return ParameterTag(key + "_ub")


key_alpha_relu_lb = ParameterTag("alpha_relu_lb")
key_alpha_relu_ub = ParameterTag("alpha_relu_ub")


def key_alpha_relu(compute_upper_bound: bool) -> ParameterTag:
    return key_alpha_relu_ub if compute_upper_bound else key_alpha_relu_lb


key_beta_lb = ParameterTag("beta_lb")
key_beta_ub = ParameterTag("beta_ub")


def key_beta(compute_upper_bound: bool) -> ParameterTag:
    return key_beta_ub if compute_upper_bound else key_beta_lb


key_prima_lb = ParameterTag("prima_lb")
key_prima_ub = ParameterTag("prima_ub")


def key_prima(compute_upper_bound: bool) -> ParameterTag:
    return key_prima_ub if compute_upper_bound else key_prima_lb


LayerTag = NewType("LayerTag", int)  # tag of a layer


def layer_tag(module: AbstractModule) -> LayerTag:
    return LayerTag(id(module))


NodeIndex = Tuple[int, ...]


@dataclass(eq=True, frozen=True)
class NodeTag(ABC):
    layer: LayerTag
    index: NodeIndex


# QueryTag = NewType(
#     "QueryTag", int
# )  # tag uniquely identifying a query relative to a network (bound min_x query_i*f(x), where f is derived from the network)

# def query_tag(
#     module: AbstractModule,
# ) -> QueryTag:  # in case we need to track one query per layer
#     return QueryTag(id(module))

# def query_tag_for_neuron(
#     module: AbstractModule, neuron_index: Tuple[int, ...]
# ) -> QueryTag:  # in case we need to track one query per neuron
#     return QueryTag(id(module))


# def layer_from_query_tag(
#     query_id: QueryTag,
# ) -> LayerTag:  # TODO: make attribute of QueryTag instead?
#     return LayerTag(QueryTag(query_id))


@dataclass(eq=True, frozen=True)
class QueryTag:
    pass


@dataclass(eq=True, frozen=True)
class SingleQueryTag(QueryTag):
    layer: LayerTag


@dataclass(eq=True, frozen=True)
class NeuronWiseQueryTag(QueryTag):
    node: NodeTag


def query_tag(
    module: AbstractModule,
) -> QueryTag:  # in case we need to track one query per layer
    return SingleQueryTag(layer=layer_tag(module))


def query_tag_for_neuron(
    module: AbstractModule, neuron_index: Tuple[int, ...]
) -> QueryTag:  # in case we need to track one query per neuron
    return NeuronWiseQueryTag(node=NodeTag(layer=layer_tag(module), index=neuron_index))


def layer_from_query_tag(
    query_id: QueryTag,
) -> LayerTag:  # TODO: make attribute of QueryTag instead?
    # TODO
    if isinstance(query_id, SingleQueryTag):
        return query_id.layer
    elif isinstance(query_id, NeuronWiseQueryTag):
        return query_id.node.layer
    else:
        raise Exception("unsupported query tag '" + str(type(query_id)) + "'")
