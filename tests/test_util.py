from typing import Sequence, Tuple

import torch
from bunch import Bunch
from torch import Tensor

from src.abstract_layers.abstract_linear import Linear
from src.abstract_layers.abstract_network import AbstractNetwork
from src.abstract_layers.abstract_relu import ReLU
from src.abstract_layers.abstract_sequential import Sequential
from src.branch_and_bound import BranchAndBound
from src.mn_bab_optimizer import MNBabOptimizer
from src.mn_bab_shape import MN_BaB_Shape

MNIST_INPUT_DIM = (1, 28, 28)
CIFAR10_INPUT_DIM = (3, 32, 32)

MNIST_FC_DATA_TEST_CONFIG = Bunch(
    {
        "input_dim": [784],
        "eps": 0.01,
    }
)

MNIST_CONV_DATA_TEST_CONFIG = Bunch(
    {
        "input_dim": MNIST_INPUT_DIM,
        "eps": 0.01,
        "normalization_means": [0.1307],
        "normalization_stds": [0.3081],
    }
)

CIFAR10_CONV_DATA_TEST_CONFIG = Bunch(
    {
        "input_dim": CIFAR10_INPUT_DIM,
        "eps": 0.01,
        "normalization_means": [0.4914, 0.4822, 0.4465],
        "normalization_stds": [0.2023, 0.1994, 0.2010],
    }
)

DEFAULT_BRANCHING_CONFIG = {
    "method": "babsr",
    "use_prima_contributions": False,
    "use_optimized_slopes": False,
    "use_beta_contributions": False,
    "propagation_effect_mode": "bias",
    "use_indirect_effect": False,
    "reduce_op": "min",
    "use_abs": True,
    "use_cost_adjusted_scores": False,
}


def toy_net() -> AbstractNetwork:
    """
    Running example of the DeepPoly paper:
    https://files.sri.inf.ethz.ch/website/papers/DeepPoly.pdf
    """
    linear1 = Linear(2, 2)
    relu1 = ReLU((2,))
    linear2 = Linear(2, 2)
    relu2 = ReLU((2,))
    linear3 = Linear(2, 2)
    linear_out = Linear(2, 1)

    linear1.weight.data = torch.tensor([[1.0, 1.0], [1.0, -1.0]])
    linear1.bias.data = torch.zeros(2)

    linear2.weight.data = torch.tensor([[1.0, 1.0], [1.0, -1.0]])
    linear2.bias.data = torch.zeros(2)

    linear3.weight.data = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
    linear3.bias.data = torch.tensor([1.0, 0.0])

    linear_out.weight.data = torch.tensor([[1.0, -1.0]])
    linear_out.bias.data = torch.zeros(1)

    return AbstractNetwork(
        Sequential([linear1, relu1, linear2, relu2, linear3, linear_out]).layers
    )


def get_deep_poly_bounds(
    network: AbstractNetwork,
    input_lb: Tensor,
    input_ub: Tensor,
    use_dependence_sets: bool = False,
    use_early_termination: bool = False,
) -> Tuple[Tensor, Tensor]:
    abstract_shape = MN_BaB_Shape.construct_to_bound_all_outputs(
        input_lb.device, network.output_dim
    )
    output_shape = network.get_mn_bab_shape(
        input_lb,
        input_ub,
        abstract_shape,
        use_dependence_sets=use_dependence_sets,
        use_early_termination=use_early_termination,
    )
    return output_shape.concretize(input_lb, input_ub)


def optimize_output_node_bounds_with_prima_crown(
    network: AbstractNetwork,
    output_idx: int,
    input_lb: Tensor,
    input_ub: Tensor,
    optimize_alpha: bool = False,
    optimize_prima: bool = False,
) -> Tuple[float, float]:
    optimizer = MNBabOptimizer(
        optimize_alpha=optimize_alpha,
        optimize_prima=optimize_prima,
    )

    lb_query_coef = torch.zeros(1, 1, *network.output_dim)
    lb_query_coef.data[0, 0, output_idx] = 1
    lb_bounded_subproblem = optimizer.bound_root_subproblem(
        lb_query_coef, network, input_lb, input_ub
    )
    lb = lb_bounded_subproblem.lower_bound

    ub_query_coef = torch.zeros(1, 1, *network.output_dim)
    ub_query_coef.data[0, 0, output_idx] = -1
    ub_bounded_subproblem = optimizer.bound_root_subproblem(
        ub_query_coef, network, input_lb, input_ub
    )
    ub = ub_bounded_subproblem.upper_bound
    ub = (-1) * ub
    return lb, ub


def lower_bound_output_node_with_branch_and_bound(
    network: AbstractNetwork,
    output_idx: int,
    input_lb: Tensor,
    input_ub: Tensor,
    batch_sizes: Sequence[int],
    early_stopping_threshold: float = float("inf"),
    optimize_alpha: bool = False,
    optimize_prima: bool = False,
) -> float:
    optimizer = MNBabOptimizer(
        optimize_alpha=optimize_alpha, optimize_prima=optimize_prima, beta_lr=0.1
    )
    bab = BranchAndBound(optimizer, batch_sizes, DEFAULT_BRANCHING_CONFIG, True)

    query_coef = torch.zeros(1, 1, *network.output_dim)
    query_coef.data[0, 0, output_idx] = 1

    return bab.lower_bound_property_with_branch_and_bound(
        "dummy_id",
        query_coef,
        network,
        input_lb,
        input_ub,
        early_stopping_threshold,
    )
