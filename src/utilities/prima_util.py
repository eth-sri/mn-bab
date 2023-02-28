import itertools
import multiprocessing
import sys
from enum import Enum
from typing import Callable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt  # type: ignore[import]
import numpy as np
import torch
from torch import Tensor

sys.path.insert(0, "ELINA/python_interface/")
from ELINA.python_interface.fconv import (  # type: ignore[import] # noqa: 401
    fkrelu,
    generate_sparse_cover,
)
from src.utilities.config import PrimaGroupRandomness  # noqa: 401


class ActivationType(Enum):
    ReLU = "ReLU"
    Sigmoid = "SIgmoid"
    Tanh = "Tanh"


# adapted from https://github.com/mnmueller/eran/blob/master/tf_verify/krelu.py
# commit hash: d57a2547a3ca3edfc2e2958f2cdac11a816f8a9c

global R_SEED
R_SEED = 42**2


class KAct:
    k: int
    input_hrep: np.ndarray
    varsid: Optional[Tuple[int, ...]]

    def __init__(
        self,
        activation_type: ActivationType,
        input_hrep: np.ndarray,
        approx: bool = True,
    ) -> None:
        assert activation_type in [
            ActivationType.ReLU,
            ActivationType.Tanh,
            ActivationType.Sigmoid,
        ]
        self.k = len(input_hrep[0]) - 1
        self.input_hrep = input_hrep
        self.varsid = None

        if activation_type == ActivationType.ReLU:
            if approx:
                self.cons = fkrelu(self.input_hrep)
            else:
                assert False, "not implemented"
                # self.cons = krelu_with_cdd(self.input_hrep)
        elif not approx:
            assert False, "not implemented"
        elif activation_type == ActivationType.Tanh:
            assert False, "not implemented"
            # self.cons = ftanh_orthant(self.input_hrep)
        else:
            assert False, "not implemented"
            # self.cons = fsigm_orthant(self.input_hrep)


class MakeKAct:  # (can't pickle closure)
    activation_type: ActivationType

    def __init__(self, activation_type: ActivationType):
        self.activation_type = activation_type

    def __call__(self, input_hrep: np.ndarray, approx: bool = True) -> KAct:
        return KAct(self.activation_type, input_hrep, approx)


def make_kactivation_obj(
    activation_type: ActivationType,
) -> MakeKAct:
    return MakeKAct(activation_type)


def sparse_heuristic_with_cutoff(
    length: int,
    lb: np.ndarray,
    ub: np.ndarray,
    sparse_n: int,
    K: int = 3,
    s: int = -2,
    max_unstable_nodes_considered_per_layer: int = 1000,
    min_relu_transformer_area_to_be_considered: float = 0.05,
) -> Tuple[Sequence[Tuple[int, ...]], Sequence[int]]:
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
    vars_above_cutoff_return = vars_above_cutoff

    kact_args: List[Tuple[int, ...]] = []
    while len(vars_above_cutoff) > 0 and sparse_n >= K:
        grouplen = min(sparse_n, len(vars_above_cutoff))
        group = tuple(vars_above_cutoff[:grouplen])
        vars_above_cutoff = vars_above_cutoff[grouplen:]
        if grouplen <= K:
            kact_args.append(group)
        elif K > 2:
            sparsed_combs: Sequence[Sequence[int]] = generate_sparse_cover(
                n=grouplen, k=K, s=s
            )
            for comb in sparsed_combs:
                kact_args.append(tuple(group[i] for i in comb))
        elif K == 2:
            raise RuntimeError("K=2 is not supported")

    # Edited: removed since those constraints are already handled
    # Also just apply 1-relu for every var.
    # for var in all_vars:
    #    kact_args.append([var])

    return kact_args, vars_above_cutoff_return


def random_group_augmentation(
    groups_multi_neuron: Sequence[Tuple[int, ...]],
    single_neurons: np.ndarray,
    K: int,
    sparse_n: int,
    random_prima_groups: PrimaGroupRandomness = PrimaGroupRandomness.none,
    prima_sparsity_factor: float = 1.0,
) -> Sequence[Tuple[int, ...]]:
    global R_SEED
    R_SEED += 1
    np.random.seed(R_SEED)
    n_cons_new = int(np.ceil(len(groups_multi_neuron) * prima_sparsity_factor))

    groups: Sequence[Tuple[int, ...]]
    if random_prima_groups == PrimaGroupRandomness.only:
        groups_list: List[Tuple[int, ...]] = []
        while len(groups_list) < n_cons_new:
            new_group = tuple(
                single_neurons[
                    np.random.choice(
                        len(single_neurons),
                        min(K, len(single_neurons)),
                        replace=False,
                    )
                ]
            )
            if new_group not in groups_list:
                groups_list.append(new_group)
        groups = groups_list
    elif random_prima_groups == PrimaGroupRandomness.augment:
        groups_list = list(groups_multi_neuron)
        if len(single_neurons) > sparse_n:
            n_cons_new = min(
                n_cons_new,
                len(groups_multi_neuron)
                + (len(single_neurons) ** 2 - sparse_n**2) // 2,
            )
            while len(groups_list) < n_cons_new:
                new_group_idx = np.random.choice(
                    len(single_neurons),
                    min(K, len(single_neurons)),
                    replace=False,
                )
                if len(np.unique(new_group_idx // sparse_n)) == 1:
                    continue
                new_group = tuple(single_neurons[new_group_idx])
                if new_group not in groups_list:
                    groups_list.append(new_group)
        groups = groups_list
    elif random_prima_groups == PrimaGroupRandomness.none:
        n_cons_new = min(n_cons_new, len(groups_multi_neuron))
        sparse_idx = np.random.choice(
            len(groups_multi_neuron), n_cons_new, replace=False
        )
        groups = [groups_multi_neuron[idx] for idx in sparse_idx]
    else:
        assert False, "Unsupported PrimaGroupRandomness." + random_prima_groups.value
    return groups


def encode_kactivation_cons(
    input_lb: Tensor,
    input_ub: Tensor,
    activation_type: ActivationType,
    sparse_n: int,
    intermediate_bounds_callback: Callable[[Tensor], Tuple[Tensor, Tensor]],
    K: int,
    s: int,
    approx: bool,
    numproc: int,
    max_number_of_parallel_input_constraint_queries: int,
    max_unstable_nodes_considered_per_layer: int,
    min_relu_transformer_area_to_be_considered: float,
    fraction_of_constraints_to_keep: float,
    random_prima_groups: PrimaGroupRandomness = PrimaGroupRandomness.none,
    prima_sparsity_factor: float = 1.0,
) -> Optional[Sequence[Sequence[KAct]]]:
    length = np.prod(input_lb[0].shape)

    lbi = np.asarray(input_lb, dtype=np.double)
    ubi = np.asarray(input_ub, dtype=np.double)

    batch_kact_args = []
    for lb, ub in zip(lbi, ubi):
        if activation_type == ActivationType.ReLU:
            groups_multi_neuron, single_neurons_seq = sparse_heuristic_with_cutoff(
                length,
                lb.flatten(),
                ub.flatten(),
                sparse_n,
                K=K,
                s=s,
                max_unstable_nodes_considered_per_layer=max_unstable_nodes_considered_per_layer,
                min_relu_transformer_area_to_be_considered=min_relu_transformer_area_to_be_considered,
            )
            single_neurons = np.array(single_neurons_seq)  # TODO: why?
            if (
                random_prima_groups != PrimaGroupRandomness.none
                or prima_sparsity_factor < 1
            ):
                groups = random_group_augmentation(
                    groups_multi_neuron,
                    single_neurons,
                    K,
                    sparse_n,
                    random_prima_groups,
                    prima_sparsity_factor,
                )
            else:
                groups = groups_multi_neuron
            batch_kact_args.append(groups)

    if not any((kact_args) for kact_args in batch_kact_args):
        return None

    activation_type = activation_type

    batch_input_hrep_array = build_octahedron_input_constraints_in_batch(
        input_lb=input_lb,
        input_ub=input_ub,
        batch_of_node_groups=batch_kact_args,
        n_nodes_in_layer=length,
        activation_type=activation_type,
        intermediate_bounds_callback=intermediate_bounds_callback,
        max_number_of_parallel_input_constraint_queries=max_number_of_parallel_input_constraint_queries,
    )

    make_kact = make_kactivation_obj(activation_type)

    if numproc == 1:
        batch_kact_results_list = []
        for input_hrep_array in batch_input_hrep_array:
            batch_kact_results_list.append(
                [
                    make_kact(inp, app)
                    for (inp, app) in zip(
                        input_hrep_array, len(input_hrep_array) * [approx]
                    )
                ]
            )
    elif numproc > 1:
        with multiprocessing.Pool(numproc) as pool:
            batch_kact_results_list = []
            for input_hrep_array in batch_input_hrep_array:
                batch_kact_results_list.append(
                    pool.starmap(
                        make_kact,
                        zip(input_hrep_array, len(input_hrep_array) * [approx]),
                    )
                )

    batch_kact_results: Sequence[Sequence[KAct]] = batch_kact_results_list

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

            assert group.varsid is not None
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


def plot_neuron_groups(
    groups: Sequence[Tuple[int]], single_neurons: Sequence[int]
) -> None:
    neuron_ids = {v: i for i, v in enumerate(single_neurons)}
    adj = np.zeros((len(single_neurons), len(single_neurons)))
    for group in groups:
        ids = [neuron_ids[x] for x in group]
        for idx in itertools.combinations(ids, 2):
            adj[idx[0], idx[1]] += 1
            adj[idx[1], idx[0]] += 1
    plt.imshow(adj)
    plt.show()


def _relu_as_last_layer(lbi: np.ndarray) -> bool:
    """
    For networks with a ReLU as the last layer, we want to keep the single neuron constraints.
    We hypothesize that just optimizing lower bound slopes is not equivalent to the triangle
    relaxation for the very last layer.
    TODO ATTENTION: this would need to be adapted if networks with a different number of output neurons
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
    input_lb: Tensor,
    input_ub: Tensor,
    batch_of_node_groups: Sequence[Sequence[Sequence[int]]],
    n_nodes_in_layer: int,
    activation_type: ActivationType,  # TODO: why is this here?
    intermediate_bounds_callback: Callable[
        [Tensor],
        Tuple[Tensor, Tensor],
    ],
    max_number_of_parallel_input_constraint_queries: int,
) -> Sequence[Sequence[np.ndarray]]:

    batch_size = len(batch_of_node_groups)

    # total number of constraints with at least two non-zero coefficients, ignoring redundant onesunder multiplication with -1:
    total_n_at_least_two_reduced = max(
        sum(
            (3 ** len(node_indices) - 1) // 2 - len(node_indices)
            for node_indices in node_groups
        )
        for node_groups in batch_of_node_groups
    )
    query_coef = torch.zeros(batch_size, total_n_at_least_two_reduced, n_nodes_in_layer)

    batch_input_octahedron_constraints_list = []
    batch_bound_masks_list = []
    for batch_index, node_groups in enumerate(batch_of_node_groups):
        input_octahedron_constraints_list = []
        bound_masks_list = []

        offset = 0
        for node_indices in node_groups:
            n_nodes_in_group = len(node_indices)
            n_upper_bounds_in_group = (3**n_nodes_in_group - 1) // 2
            n_at_least_two_reduced = n_upper_bounds_in_group - n_nodes_in_group
            input_octahedron_constraint_matrix_of_group = np.zeros(
                (2 * n_upper_bounds_in_group, n_nodes_in_group + 1)
            )
            octahedron_coefs = torch.cartesian_prod(
                *(
                    (torch.tensor([1.0, 0.0, -1.0]),) * n_nodes_in_group
                )  # (Those are all coefficients, we will filter redundant ones.)
            ).view(3**n_nodes_in_group, -1)
            # assert (octahedron_coefs[:n_upper_bounds_in_group, :] == -octahedron_coefs[n_upper_bounds_in_group+1:,:].flip(dims=(0,))).all()
            reduced_octahedron_coefs = octahedron_coefs[
                :n_upper_bounds_in_group, :
            ]  # first, get rid of coefficients that are redundant under multiplication with -1

            # assert (octahedron_coefs[n_upper_bounds_in_group, :] == 0).all()
            # octahedron_coefs = octahedron_coefs[~(octahedron_coefs==0).all(1)]  # equivalent to next line
            octahedron_coefs = torch.cat(
                (
                    octahedron_coefs[:n_upper_bounds_in_group],
                    octahedron_coefs[n_upper_bounds_in_group + 1 :],
                ),
                dim=0,
            )  # filter out zero

            assert octahedron_coefs.shape == (
                2 * n_upper_bounds_in_group,
                n_nodes_in_group,
            )

            lb_coef_mask = (reduced_octahedron_coefs != 0).sum(
                1
            ) == 1  # lower bound coefficients (TODO: better way to compute the indices?)
            ub_coef_mask = lb_coef_mask.flip(dims=(0,))  # upper bound coefficients

            single_bound_coef_mask = torch.cat((lb_coef_mask, ub_coef_mask), dim=0)
            multi_bound_coef_mask = ~single_bound_coef_mask

            reduced_octahedron_coefs = reduced_octahedron_coefs[
                ~lb_coef_mask, :
            ]  # now, drop coefficients involving only a single value

            assert reduced_octahedron_coefs.shape == (
                n_at_least_two_reduced,
                n_nodes_in_group,
            )

            # lb_coefs = torch.eye(n_nodes_in_group)
            # assert (octahedron_coefs[single_bound_coef_mask] == torch.cat((lb_coefs, -lb_coefs.flip(dims=(0,))))).all()
            # assert (octahedron_coefs[multi_bound_coef_mask] == torch.cat((reduced_octahedron_coefs, -reduced_octahedron_coefs.flip(dims=(0,))), dim=0)).all()

            input_octahedron_constraint_matrix_of_group[:, 1:] = octahedron_coefs.cpu()
            query_coef.data[
                batch_index,
                offset : offset + n_at_least_two_reduced,
                node_indices,
            ] = reduced_octahedron_coefs

            input_octahedron_constraints_list.append(
                input_octahedron_constraint_matrix_of_group
            )
            bound_masks_list.append(
                (
                    np.array(single_bound_coef_mask),
                    np.array(multi_bound_coef_mask),
                )
            )
            offset += n_at_least_two_reduced
        batch_input_octahedron_constraints_list.append(
            input_octahedron_constraints_list
        )
        batch_bound_masks_list.append(bound_masks_list)

    # only do number_of_nodes_in_starting_layer many queries at a time
    number_of_queries = query_coef.shape[1]
    batch_intermediate_lb = torch.zeros(batch_size, number_of_queries)
    batch_intermediate_ub = torch.zeros(batch_size, number_of_queries)
    offset = 0
    while offset < number_of_queries:
        query_coef_slice = query_coef[
            :, offset : offset + max_number_of_parallel_input_constraint_queries, :
        ]

        intermediate_lb, intermediate_ub = intermediate_bounds_callback(
            query_coef_slice
        )

        batch_intermediate_lb[
            :, offset : offset + max_number_of_parallel_input_constraint_queries
        ] = intermediate_lb

        batch_intermediate_ub[
            :, offset : offset + max_number_of_parallel_input_constraint_queries
        ] = intermediate_ub
        offset += max_number_of_parallel_input_constraint_queries

    for batch_index, (intermediate_lb, intermediate_ub) in enumerate(
        zip(batch_intermediate_lb, batch_intermediate_ub)
    ):
        all_deep_poly_lower_bounds = np.array(intermediate_lb)
        all_deep_poly_upper_bounds = np.array(intermediate_ub)
        all_input_lower_bounds = np.array(input_lb[batch_index].flatten())
        all_input_upper_bounds = np.array(input_ub[batch_index].flatten())

        offset = 0
        for i, (input_octahedron_constraint_matrix_of_group) in enumerate(
            batch_input_octahedron_constraints_list[batch_index]
        ):
            node_indices_np = np.array(batch_of_node_groups[batch_index][i])
            n_nodes_in_group = len(node_indices_np)
            n_upper_bounds_in_group = (3**n_nodes_in_group - 1) // 2
            n_at_least_two_reduced = n_upper_bounds_in_group - n_nodes_in_group

            (
                single_bound_coef_mask_np,
                multi_bound_coef_mask_np,
            ) = batch_bound_masks_list[batch_index][i]

            input_octahedron_constraint_matrix_of_group[
                single_bound_coef_mask_np, 0
            ] = np.concatenate(
                (
                    -all_input_lower_bounds[node_indices_np],
                    np.flipud(all_input_upper_bounds[node_indices_np]),
                ),
                axis=0,
            )
            input_octahedron_constraint_matrix_of_group[
                multi_bound_coef_mask_np, 0
            ] = np.concatenate(
                (
                    -all_deep_poly_lower_bounds[
                        offset : offset + n_at_least_two_reduced
                    ],
                    np.flipud(
                        all_deep_poly_upper_bounds[
                            offset : offset + n_at_least_two_reduced
                        ]
                    ),
                ),
                axis=0,
            )
            offset += n_at_least_two_reduced

    return batch_input_octahedron_constraints_list
