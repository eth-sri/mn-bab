# mypy: ignore-errors
import multiprocessing
import sys
from typing import Callable, Sequence

import numpy as np
import torch
from torch import Tensor

sys.path.insert(0, "ELINA/python_interface/")
from ELINA.python_interface.fconv import fkrelu, generate_sparse_cover  # noqa: E402

# adapted from https://github.com/mnmueller/eran/blob/master/tf_verify/krelu.py
# commit hash: d57a2547a3ca3edfc2e2958f2cdac11a816f8a9c


class KAct:
    def __init__(self, input_hrep, approx=True):
        self.k = len(input_hrep[0]) - 1
        self.input_hrep = np.array(input_hrep)
        self.varsid = None

        if approx:
            self.cons = fkrelu(self.input_hrep)
        else:
            assert False, "not implemented"
            # self.cons = krelu_with_cdd(self.input_hrep)


def make_kactivation_obj(input_hrep, approx=True):
    return KAct(input_hrep, approx)


def sparse_heuristic_with_cutoff(
    length,
    lb,
    ub,
    sparse_n,
    K=3,
    s=-2,
    max_unstable_nodes_considered_per_layer=1000,
    min_relu_transformer_area_to_be_considered=0.05,
):
    assert length == len(lb) == len(ub)

    all_vars = [i for i in range(length) if lb[i] < 0 < ub[i]]
    areas = {var: -lb[var] * ub[var] for var in all_vars}

    assert len(all_vars) == len(areas)
    # Sort vars by descending area
    all_vars = sorted(all_vars, key=lambda var: -areas[var])

    vars_above_cutoff = [
        i
        for i in all_vars[:max_unstable_nodes_considered_per_layer]
        if areas[i] >= min_relu_transformer_area_to_be_considered
    ]
    # n_vars_above_cutoff = len(vars_above_cutoff)

    kact_args = []
    while len(vars_above_cutoff) > 0 and sparse_n >= K:
        grouplen = min(sparse_n, len(vars_above_cutoff))
        group = vars_above_cutoff[:grouplen]
        vars_above_cutoff = vars_above_cutoff[grouplen:]
        if grouplen <= K:
            kact_args.append(group)
        elif K > 2:
            sparsed_combs = generate_sparse_cover(grouplen, K, s=s)
            for comb in sparsed_combs:
                kact_args.append(tuple([group[i] for i in comb]))
        elif K == 2:
            raise RuntimeError("K=2 is not supported")

    # Edited: removed since those constraints are already handled
    # Also just apply 1-relu for every var.
    # for var in all_vars:
    #    kact_args.append([var])

    return kact_args


def encode_kactivation_cons(
    lbi,
    ubi,
    sparse_n,
    intermediate_bounds_callback,
    K,
    s,
    approx,
    numproc,
    max_number_of_parallel_input_constraint_queries,
    max_unstable_nodes_considered_per_layer,
    min_relu_transformer_area_to_be_considered,
    fraction_of_constraints_to_keep,
):
    length = np.prod(lbi[0].shape)

    lbi = np.asarray(lbi, dtype=np.double)
    ubi = np.asarray(ubi, dtype=np.double)

    batch_kact_args = []
    for lb, ub in zip(lbi, ubi):
        batch_kact_args.append(
            [
                group
                for group in sparse_heuristic_with_cutoff(
                    length,
                    lb.flatten(),
                    ub.flatten(),
                    sparse_n,
                    K=K,
                    s=s,
                    max_unstable_nodes_considered_per_layer=max_unstable_nodes_considered_per_layer,
                    min_relu_transformer_area_to_be_considered=min_relu_transformer_area_to_be_considered,
                )
                if len(group) > 1
            ]
        )

    if not any([(kact_args) for kact_args in batch_kact_args]):
        return None

    batch_input_hrep_array = build_octahedron_input_constraints_in_batch(
        batch_kact_args,
        length,
        intermediate_bounds_callback,
        max_number_of_parallel_input_constraint_queries,
    )

    if numproc == 1:
        batch_kact_results = []
        for input_hrep_array in batch_input_hrep_array:
            batch_kact_results.append(
                [
                    make_kactivation_obj(inp, app)
                    for (inp, app) in zip(
                        input_hrep_array, len(input_hrep_array) * [approx]
                    )
                ]
            )
    elif numproc > 1:
        with multiprocessing.Pool(numproc) as pool:
            batch_kact_results = []
            for input_hrep_array in batch_input_hrep_array:
                batch_kact_results.append(
                    pool.starmap(
                        make_kactivation_obj,
                        zip(input_hrep_array, len(input_hrep_array) * [approx]),
                    )
                )

    output_lbi = np.clip(lbi, a_min=0, a_max=None)
    output_ubi = ubi
    batch_worst_violations = []
    for batch_index, (kact_results, kact_args) in enumerate(
        zip(batch_kact_results, batch_kact_args)
    ):
        gid = 0
        for inst in kact_results:
            varsid = kact_args[gid]
            inst.varsid = varsid
            gid = gid + 1

        worst_violations_per_group = []
        for group in kact_results:
            output_constraints = group.cons[:, group.k + 1 :]
            input_constraints = group.cons[:, 1 : group.k + 1]
            const_constraints = group.cons[:, 0]

            group_indices = list(group.varsid)
            worst_violations_per_group.append(
                (
                    const_constraints
                    + np.sum(
                        np.where(
                            output_constraints > 0,
                            output_lbi[batch_index].flatten()[group_indices],
                            output_ubi[batch_index].flatten()[group_indices],
                        )
                        * output_constraints,
                        axis=1,
                    )
                    + np.sum(
                        np.where(
                            input_constraints > 0,
                            lbi[batch_index].flatten()[group_indices],
                            ubi[batch_index].flatten()[group_indices],
                        )
                        * input_constraints,
                        axis=1,
                    )
                ).tolist()
            )
        batch_worst_violations.append(worst_violations_per_group)

    if not _relu_as_last_layer(lbi):
        batch_kact_results = _filter_constraints(
            batch_kact_results, batch_worst_violations, fraction_of_constraints_to_keep
        )

    return batch_kact_results


def _relu_as_last_layer(lbi: np.ndarray) -> bool:
    """
    For networks with a ReLU as the last layer, we want to keep the single neuron constraints.
    We hypothesize that just optimizing lower bound slopes is not equivalent to the triangle
    relaxation for the very last layer.
    ATTENTION: this would need to be adapted if networks with a different number of output neurons
    are considered.
    """
    return lbi.shape[1] == 10


def _filter_constraints(
    batch_kact_results: Sequence[Sequence[KAct]],
    batch_worst_violations: Sequence[Sequence[Sequence[float]]],
    percentage_to_keep: float,
) -> Sequence[Sequence[KAct]]:
    for kact_results, worst_violations_per_group in zip(
        batch_kact_results, batch_worst_violations
    ):
        worst_violation_treshold = np.quantile(
            [
                violation
                for worst_violations_in_group in worst_violations_per_group
                for violation in worst_violations_in_group
            ],
            percentage_to_keep,
        )
        for group, worst_violations in zip(kact_results, worst_violations_per_group):
            output_constraints = group.cons[:, group.k + 1 :]
            input_constraints = group.cons[:, 1 : group.k + 1]

            single_neuron_constraints = []
            for row_index, (output_constraint, input_constraint) in enumerate(
                zip(output_constraints, input_constraints)
            ):
                non_zero_columns_in_output_constraints = output_constraint.nonzero()[0]
                non_zero_columns_in_input_constraints = input_constraint.nonzero()[0]
                if (
                    len(non_zero_columns_in_output_constraints) == 0
                    or len(non_zero_columns_in_input_constraints) == 0
                    or (
                        (
                            np.array_equal(
                                non_zero_columns_in_output_constraints,
                                non_zero_columns_in_input_constraints,
                            )
                        )
                        and (len(non_zero_columns_in_output_constraints) == 1)
                    )
                ):
                    single_neuron_constraints.append(row_index)

            constraints_not_satisfying_violation_threshold = (
                np.argwhere(
                    [
                        violation >= worst_violation_treshold
                        for violation in worst_violations
                    ]
                )
                .flatten()
                .tolist()
            )
            constraints_to_delete = list(
                set(single_neuron_constraints)
                | set(constraints_not_satisfying_violation_threshold)
            )
            group.cons = np.delete(group.cons, constraints_to_delete, axis=0)

    return batch_kact_results


def build_octahedron_input_constraints_in_batch(
    batch_of_node_groups: Sequence[Sequence[Sequence[int]]],
    n_nodes_in_layer: int,
    intermediate_bounds_callback: Callable[
        [Tensor],
        Sequence[float],
    ],
    max_number_of_parallel_input_constraint_queries: int,
) -> Sequence[Sequence[np.ndarray]]:
    batch_size = len(batch_of_node_groups)

    max_n_total_upper_bounds = max(
        [
            sum([(3 ** len(node_indices) - 1) for node_indices in node_groups])
            for node_groups in batch_of_node_groups
        ]
    )

    batch_input_octahedron_constraints_list = []
    query_coef = torch.zeros(batch_size, max_n_total_upper_bounds, n_nodes_in_layer)
    for batch_index, node_groups in enumerate(batch_of_node_groups):
        input_octahedron_constraints_list = []

        offset = 0
        for node_indices in node_groups:
            n_nodes_in_group = len(node_indices)
            n_upper_bounds_in_group = 3 ** n_nodes_in_group - 1
            input_octahedron_constraint_matrix_of_group = np.zeros(
                (n_upper_bounds_in_group, n_nodes_in_group + 1)
            )
            octahedron_coefs = torch.cartesian_prod(
                *((torch.tensor([-1.0, 0.0, 1.0]),) * n_nodes_in_group)
            ).view(n_upper_bounds_in_group + 1, -1)
            octahedron_coefs = octahedron_coefs[(octahedron_coefs != 0).any(1), :]
            input_octahedron_constraint_matrix_of_group[0:, 1:] = (
                -1
            ) * octahedron_coefs.cpu()
            query_coef.data[
                (
                    batch_index,
                    slice(offset, offset + n_upper_bounds_in_group),
                    node_indices,
                )
            ] = octahedron_coefs

            input_octahedron_constraints_list.append(
                input_octahedron_constraint_matrix_of_group
            )
            offset += n_upper_bounds_in_group
        batch_input_octahedron_constraints_list.append(
            input_octahedron_constraints_list
        )

    # only do number_of_nodes_in_starting_layer many queries at a time
    number_of_queries = query_coef.shape[1]
    batch_intermediate_ub = torch.zeros(batch_size, number_of_queries)
    offset = 0
    while offset < number_of_queries:
        query_coef_slice = query_coef[
            :, offset : offset + max_number_of_parallel_input_constraint_queries, :
        ]
        __, intermediate_ub = intermediate_bounds_callback(query_coef_slice)
        batch_intermediate_ub[
            :, offset : offset + max_number_of_parallel_input_constraint_queries
        ] = intermediate_ub
        offset += max_number_of_parallel_input_constraint_queries

    for batch_index, intermediate_ub in enumerate(batch_intermediate_ub):
        all_deep_poly_upper_bounds = intermediate_ub.squeeze().tolist()
        offset = 0
        for i, (input_octahedron_constraint_matrix_of_group) in enumerate(
            batch_input_octahedron_constraints_list[batch_index]
        ):
            n_nodes_in_group = len(batch_of_node_groups[batch_index][i])
            n_upper_bounds_in_group = 3 ** n_nodes_in_group - 1
            input_octahedron_constraint_matrix_of_group[
                :, 0
            ] = all_deep_poly_upper_bounds[offset : offset + n_upper_bounds_in_group]
            offset += n_upper_bounds_in_group
    return batch_input_octahedron_constraints_list
