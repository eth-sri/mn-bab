from typing import Callable, List, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from src.utilities.config import PrimaHyperparameters
from src.utilities.prima_util import ActivationType, KAct, encode_kactivation_cons


def get_prima_constraints(
    input_lb: Tensor,
    input_ub: Tensor,
    activation_type: ActivationType,
    prima_hyperparameters: PrimaHyperparameters,
    intermediate_bounds_callback: Callable[[Tensor], Tuple[Tensor, Tensor]],
    batch_size: int,
    layer_shape: Tuple[int, ...],
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    The PRIMA constraints are of the form:
     output_var_coefs @ layer_output + input_var_coefs @ layer_input + const_coefs @ 1 <= 0

    :param input_lb: input lower bounds for neurons in current layer
    :param input_ub: input uppper bounds for neurons in current layer
    :returns:
        output_var_coefs
        input_var_coefs
        const_coefs
    """

    batch_prima_constraints = encode_kactivation_cons(
        input_lb=input_lb,
        input_ub=input_ub,
        activation_type=activation_type,
        # TODO: just pass through prima_hyperparameters
        sparse_n=prima_hyperparameters.sparse_n,
        intermediate_bounds_callback=intermediate_bounds_callback,
        K=prima_hyperparameters.K,
        s=prima_hyperparameters.s,
        approx=True,
        numproc=prima_hyperparameters.num_proc_to_compute_constraints,
        max_number_of_parallel_input_constraint_queries=prima_hyperparameters.max_number_of_parallel_input_constraint_queries,
        max_unstable_nodes_considered_per_layer=prima_hyperparameters.max_unstable_nodes_considered_per_layer,
        min_relu_transformer_area_to_be_considered=prima_hyperparameters.min_relu_transformer_area_to_be_considered,
        fraction_of_constraints_to_keep=prima_hyperparameters.fraction_of_constraints_to_keep,
        random_prima_groups=prima_hyperparameters.random_prima_groups,
        prima_sparsity_factor=prima_hyperparameters.prima_sparsity_factor,
    )

    prima_constraints_empty = True
    if batch_prima_constraints is not None:
        for batch_elem in batch_prima_constraints:
            if not prima_constraints_empty:
                break
            for kact in batch_elem:
                if not prima_constraints_empty:
                    break
                prima_constraints_empty = kact.cons.shape[0] == 0

    if not batch_prima_constraints or prima_constraints_empty:
        n_prima_constraints = 0
        output_coefs = torch.zeros(
            batch_size, np.prod(layer_shape), n_prima_constraints
        )
        input_coefs = torch.zeros(batch_size, np.prod(layer_shape), n_prima_constraints)
        const_coefs = torch.zeros(batch_size, 1, n_prima_constraints)
        return output_coefs, input_coefs, const_coefs

    return _build_sparse_prima_coefficient_matrix(batch_prima_constraints, layer_shape)


def _refine_bounds_for_candidate_unstable_neurons(
    lb: Tensor,
    ub: Tensor,
    batch_size: int,
    layer_shape: Tuple[int, ...],
    intermediate_bounds_callback: Callable[[Tensor], Tuple[Tensor, Tensor]],
) -> Tuple[Tensor, Tensor]:
    candidate_unstable_neuron_indices = torch.nonzero(
        (lb < 0) & (ub > 0), as_tuple=True
    )
    if candidate_unstable_neuron_indices[0].numel() == 0:
        return lb, ub

    n_unstable_neurons_per_batch_element = torch.bincount(
        candidate_unstable_neuron_indices[0]
    )
    indices_in_batch_element = [
        index
        for n_unstable_neurons in n_unstable_neurons_per_batch_element
        for index in np.arange(n_unstable_neurons)
    ]
    candidate_unstable_neuron_indices_in_batch = (
        candidate_unstable_neuron_indices[0],
        indices_in_batch_element,
        *candidate_unstable_neuron_indices[1:],
    )

    query_coef = torch.zeros(  # type: ignore[call-overload]
        batch_size,
        torch.max(n_unstable_neurons_per_batch_element),
        *layer_shape,
    )
    query_coef[candidate_unstable_neuron_indices_in_batch] = -1

    # only do number_of_nodes_in_starting_layer many queries at a time
    number_of_queries = query_coef.shape[1]
    number_of_nodes_in_starting_layer = np.prod(query_coef.shape[2:])
    lb_with_other_params = torch.zeros(batch_size, number_of_queries)
    ub_with_other_params = torch.zeros(batch_size, number_of_queries)
    offset = 0
    while offset < number_of_queries:
        query_coef_slice = query_coef[
            :, offset : offset + number_of_nodes_in_starting_layer, :
        ]
        (
            intermediate_ub_with_other_params,
            intermediate_lb_with_other_params,
        ) = intermediate_bounds_callback(query_coef_slice)
        lb_with_other_params[
            :, offset : offset + number_of_nodes_in_starting_layer
        ] = intermediate_lb_with_other_params
        ub_with_other_params[
            :, offset : offset + number_of_nodes_in_starting_layer
        ] = intermediate_ub_with_other_params
        offset += number_of_nodes_in_starting_layer

    candidate_unstable_neuron_indices_in_resulting_bounds = (
        candidate_unstable_neuron_indices[0],
        indices_in_batch_element,
    )
    lb_with_other_params = (
        -1
        * lb_with_other_params[candidate_unstable_neuron_indices_in_resulting_bounds]
        .detach()
        .cpu()
    )
    ub_with_other_params = (
        -1
        * ub_with_other_params[candidate_unstable_neuron_indices_in_resulting_bounds]
        .detach()
        .cpu()
    )
    refined_lb = lb.clone()
    refined_ub = ub.clone()

    refined_lb[candidate_unstable_neuron_indices] = torch.maximum(
        lb[candidate_unstable_neuron_indices], lb_with_other_params
    )
    refined_ub[candidate_unstable_neuron_indices] = torch.minimum(
        ub[candidate_unstable_neuron_indices], ub_with_other_params
    )

    return refined_lb, refined_ub


def _build_sparse_prima_coefficient_matrix(
    batch_prima_constraints: Sequence[Sequence[KAct]],
    layer_shape: Tuple[int, ...],
) -> Tuple[Tensor, Tensor, Tensor]:
    batch_indices: List[int] = []
    indices_within_batch_element: List[int] = []
    indices_within_layer: List[int] = []

    output_coef_values: List[float] = []
    input_coef_values: List[float] = []

    batch_size = len(batch_prima_constraints)
    max_number_of_prima_constraints = max(
        sum(constraint_group.cons.shape[0] for constraint_group in prima_constraints)
        for prima_constraints in batch_prima_constraints
    )
    const_coefs = torch.zeros(batch_size, 1, max_number_of_prima_constraints)

    for batch_index, prima_constraints in enumerate(batch_prima_constraints):
        n_prima_coefficients_in_batch_element = sum(
            len(constraint_group.varsid) * constraint_group.cons.shape[0]  # type: ignore
            for constraint_group in prima_constraints
        )
        batch_indices += [
            batch_index for __ in range(n_prima_coefficients_in_batch_element)
        ]

        offset = 0
        for constraint_group in prima_constraints:
            n_prima_constraints_in_group = constraint_group.cons.shape[0]
            node_indices = constraint_group.varsid
            group_size = len(node_indices)  # type: ignore
            indices_within_batch_element += (
                np.arange(offset, offset + n_prima_constraints_in_group)
                .repeat(group_size)
                .tolist()
            )
            indices_within_layer += np.tile(
                node_indices, n_prima_constraints_in_group
            ).tolist()

            output_coef_values += (
                constraint_group.cons[:, group_size + 1 :].ravel(order="C").tolist()
            )
            input_coef_values += (
                constraint_group.cons[:, 1 : group_size + 1].ravel(order="C").tolist()
            )

            const_coefs_of_group = torch.tensor(constraint_group.cons[:, 0])
            const_coefs[
                batch_index, 0, offset : offset + n_prima_constraints_in_group
            ] = const_coefs_of_group
            offset += n_prima_constraints_in_group

    full_indices_of_non_zero_elements = [
        batch_indices,
        indices_within_layer,
        indices_within_batch_element,
    ]

    dense_coefs_shape = (
        batch_size,
        np.prod(layer_shape),
        max_number_of_prima_constraints,
    )
    output_coefs = torch.sparse_coo_tensor(
        full_indices_of_non_zero_elements, output_coef_values, size=dense_coefs_shape  # type: ignore[arg-type]
    )
    input_coefs = torch.sparse_coo_tensor(
        full_indices_of_non_zero_elements, input_coef_values, size=dense_coefs_shape  # type: ignore[arg-type]
    )

    def _eliminate_zeros(x: Tensor) -> Tensor:
        assert x.is_sparse
        mask = x._values().nonzero()
        non_zero_values = x._values().index_select(0, mask.view(-1))
        indices_of_non_zero_values = x._indices().index_select(1, mask.view(-1))

        return torch.sparse_coo_tensor(
            indices_of_non_zero_values, non_zero_values, x.shape
        )

    output_coefs = _eliminate_zeros(output_coefs).coalesce()
    input_coefs = _eliminate_zeros(input_coefs).coalesce()

    # ELINA computes constraints s.t. coefs * vars >= 0, we want <= 0
    return (-1) * output_coefs, (-1) * input_coefs, (-1) * const_coefs
