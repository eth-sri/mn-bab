import functools
import time
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import tqdm  # type: ignore[import]
from bunch import Bunch  # type: ignore[import]
from torch import Tensor
from torch.distributions.beta import Beta

from src.abstract_domains.DP_f import DeepPoly_f, HybridZonotope
from src.abstract_layers.abstract_linear import Linear
from src.abstract_layers.abstract_network import AbstractNetwork
from src.abstract_layers.abstract_relu import ReLU
from src.abstract_layers.abstract_sequential import Sequential
from src.abstract_layers.abstract_sigmoid import Sigmoid
from src.abstract_layers.abstract_tanh import Tanh
from src.branch_and_bound import BranchAndBound
from src.concrete_layers.pad import Pad
from src.concrete_layers.split_block import SplitBlock
from src.concrete_layers.unbinary_op import UnbinaryOp
from src.milp_network import MILPNetwork
from src.mn_bab_optimizer import MNBabOptimizer
from src.mn_bab_shape import AffineForm, MN_BaB_Shape
from src.mn_bab_verifier import MNBaBVerifier
from src.state.tags import layer_tag, query_tag
from src.utilities.config import (
    MNBabOptimizerConfig,
    MNBabVerifierConfig,
    make_backsubstitution_config,
    make_optimizer_config,
    make_prima_hyperparameters,
    make_verifier_config,
)
from src.utilities.initialization import seed_everything
from src.utilities.loading.network import freeze_network, load_onnx_model, mnist_a_b
from src.utilities.queries import get_output_bound_initial_query_coef
from src.verification_subproblem import SubproblemState

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


def toy_net() -> Tuple[AbstractNetwork, Tuple[int]]:
    """
    Running example of the DeepPoly paper:
    https://files.sri.inf.ethz.ch/website/papers/DeepPoly.pdf
    """
    input_dim = (2,)

    linear1 = Linear(2, 2, bias=True, input_dim=(2,))
    relu1 = ReLU((2,))
    linear2 = Linear(2, 2, bias=True, input_dim=(2,))
    relu2 = ReLU((2,))
    linear3 = Linear(2, 2, bias=True, input_dim=(2,))
    linear_out = Linear(2, 1, bias=True, input_dim=(2,))

    linear1.weight.data = torch.tensor([[1.0, 1.0], [1.0, -1.0]])
    linear1.bias.data = torch.zeros(2)

    linear2.weight.data = torch.tensor([[1.0, 1.0], [1.0, -1.0]])
    linear2.bias.data = torch.zeros(2)

    linear3.weight.data = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
    linear3.bias.data = torch.tensor([1.0, 0.0])

    linear_out.weight.data = torch.tensor([[1.0, -1.0]])
    linear_out.bias.data = torch.zeros(1)

    return (
        AbstractNetwork(
            Sequential([linear1, relu1, linear2, relu2, linear3, linear_out]).layers
        ),
        input_dim,
    )


def toy_sig_net() -> Tuple[AbstractNetwork, Tuple[int]]:
    input_dim = (2,)

    linear1 = Linear(2, 2, bias=True, input_dim=(2,))
    sig1 = Sigmoid((2,))
    linear2 = Linear(2, 2, bias=True, input_dim=(2,))
    sig2 = Sigmoid((2,))
    linear3 = Linear(2, 2, bias=True, input_dim=(2,))
    linear_out = Linear(2, 1, bias=True, input_dim=(2,))

    linear1.weight.data = torch.tensor([[1.0, 1.0], [1.0, -1.0]])
    linear1.bias.data = torch.zeros(2)

    linear2.weight.data = torch.tensor([[1.0, 1.0], [1.0, -1.0]])
    linear2.bias.data = torch.zeros(2)

    linear3.weight.data = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
    linear3.bias.data = torch.tensor([1.0, 0.0])

    linear_out.weight.data = torch.tensor([[1.0, -1.0]])
    linear_out.bias.data = torch.zeros(1)

    return (
        AbstractNetwork(
            Sequential([linear1, sig1, linear2, sig2, linear3, linear_out]).layers
        ),
        input_dim,
    )


def toy_sig_tanh_net() -> Tuple[AbstractNetwork, Tuple[int]]:
    input_dim = (2,)

    linear1 = Linear(2, 2, bias=True, input_dim=(2,))
    sig1 = Sigmoid((2,))
    linear2 = Linear(2, 2, bias=True, input_dim=(2,))
    sig2 = Tanh((2,))
    linear3 = Linear(2, 2, bias=True, input_dim=(2,))
    linear_out = Linear(2, 1, bias=True, input_dim=(2,))

    linear1.weight.data = torch.tensor([[1.0, 1.0], [1.0, -1.0]])
    linear1.bias.data = torch.zeros(2)

    linear2.weight.data = torch.tensor([[1.0, 1.0], [1.0, -1.0]])
    linear2.bias.data = torch.zeros(2)

    linear3.weight.data = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
    linear3.bias.data = torch.tensor([1.0, 0.0])

    linear_out.weight.data = torch.tensor([[1.0, -1.0]])
    linear_out.bias.data = torch.zeros(1)

    return (
        AbstractNetwork(
            Sequential([linear1, sig1, linear2, sig2, linear3, linear_out]).layers
        ),
        input_dim,
    )


def toy_all_layer_net() -> Tuple[AbstractNetwork, Tuple[int, int, int]]:
    """
    A network that contains all currently supported layers
    """
    input_dim = (1, 5, 5)
    conv1 = nn.Conv2d(1, 3, (3, 3))  # 1x1x5x5 -> 1x3x3x3
    bn1 = nn.BatchNorm2d(3)
    flatten1 = nn.Flatten()
    relu1 = nn.ReLU()
    linear1 = nn.Linear(27, 10)
    relu2 = nn.ReLU()

    return (
        AbstractNetwork.from_concrete_module(
            nn.Sequential(*[conv1, bn1, flatten1, relu1, linear1, relu2]), input_dim
        ),
        input_dim,
    )


def toy_all_layer_net_1d() -> Tuple[AbstractNetwork, Tuple[int, ...]]:
    """
    A network that contains all currently supported layers
    """
    # Input size: 6x6
    input_size = (4,)
    layers: List[torch.nn.Module] = []
    layers += [nn.Linear(4, 10), nn.ReLU()]
    layers += [nn.Linear(10, 20), nn.ReLU()]
    layers += [nn.Linear(20, 10), nn.ReLU()]
    layers += [nn.Linear(10, 1)]

    return (
        AbstractNetwork.from_concrete_module(nn.Sequential(*layers), input_size),
        input_size,
    )


def toy_avg_pool_net() -> Tuple[AbstractNetwork, Tuple[int, int, int]]:
    """A toy network containing three avg. pool layers

    Returns:
        AbstractNetwork: AS Wrapper around the above network
    """
    input_dim = (1, 6, 6)
    # Input size: 6x6
    avg_p1 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))  # 3x3
    avg_p2 = nn.AvgPool2d(
        kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
    )  # Identity 3x3
    avg_p3 = nn.AvgPool2d(kernel_size=(2, 2), stride=(1, 1))  # 2x2
    flatten1 = nn.Flatten()

    return (
        AbstractNetwork.from_concrete_module(
            nn.Sequential(*[avg_p1, avg_p2, avg_p3, flatten1]), input_dim
        ),
        input_dim,
    )


def toy_max_pool_net() -> Tuple[AbstractNetwork, Tuple[int, int, int]]:
    """A toy network containing three max. pool layers

    Returns:
        abstract_network: AS Wrapper around the above network
    """
    input_dim = (1, 6, 6)
    # Input size: 6x6
    max_p1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))  # 3x3
    max_p2 = nn.MaxPool2d(
        kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
    )  # Identity 3x3
    max_p3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1))  # 2x2
    flatten1 = nn.Flatten()

    return (
        AbstractNetwork.from_concrete_module(
            nn.Sequential(*[max_p1, max_p2, max_p3, flatten1]), input_dim
        ),
        input_dim,
    )


def pad_toy_max_pool_net() -> Tuple[AbstractNetwork, Tuple[int, int, int]]:
    """A toy network containing three max. pool layers, none of which are the inital layer

    Returns:
        abstract_network: AS Wrapper around the above network
    """
    input_dim = (1, 10, 10)
    # Input size: 6x6
    conv_1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1))
    max_p1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))  # 3x3
    max_p2 = nn.MaxPool2d(
        kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
    )  # Identity 3x3
    max_p3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1))  # 2x2
    flatten1 = nn.Flatten()
    relu = nn.ReLU()

    return (
        AbstractNetwork.from_concrete_module(
            nn.Sequential(*[conv_1, relu, max_p1, max_p2, max_p3, flatten1]), input_dim
        ),
        input_dim,
    )


def toy_max_pool_mixed_net() -> Tuple[AbstractNetwork, Tuple[int, int, int]]:
    """A toy network containing three max. pool layers, none of which are the inital layer

    Returns:
        abstract_network: AS Wrapper around the above network
        input_size: size of the input
    """
    # Input size: 6x6
    input_size = (1, 6, 6)
    layers: List[torch.nn.Module] = []
    layers += [
        nn.Conv2d(
            in_channels=input_size[0],
            out_channels=3,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
    ]
    layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1), padding=(1, 1))]  # 6x6
    layers += [
        nn.Conv2d(in_channels=3, out_channels=2, kernel_size=(3, 3), padding=(1, 1))
    ]
    layers += [nn.ReLU()]
    layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(1, 1))]  # 4x4
    layers += [nn.Conv2d(in_channels=2, out_channels=2, kernel_size=(2, 2))]  # 2x2
    layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]  # 1x1
    layers += [nn.Flatten()]  # 2
    return (
        AbstractNetwork.from_concrete_module(nn.Sequential(*layers), input_size),
        input_size,
    )


def toy_max_pool_tiny_net() -> Tuple[AbstractNetwork, Tuple[int, int, int]]:
    """A toy network containing three max. pool layers, none of which are the inital layer

    Returns:
        abstract_network: AS Wrapper around the above network
        input_size: size of the input
    """
    # Input size: 6x6
    input_size = (1, 1, 1)
    layers: List[torch.nn.Module] = []
    layers += [nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))]  # 4x4
    layers += [nn.Flatten()]  # 4
    return (
        AbstractNetwork.from_concrete_module(nn.Sequential(*layers), input_size),
        input_size,
    )


def toy_max_avg_pool_net() -> AbstractNetwork:
    """A toy network containing three max. pool layers

    Returns:
        abstract_network: AS Wrapper around the above network
    """
    # Input size: 1x16x16
    conv1 = nn.Conv2d(1, 3, (3, 3))
    max_p1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
    relu1 = nn.ReLU()
    avg_p1 = nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    flatten1 = nn.Flatten()
    relu2 = nn.ReLU()
    linear1 = nn.Linear(147, 10)

    return AbstractNetwork.from_concrete_module(
        nn.Sequential(*[conv1, max_p1, relu1, avg_p1, flatten1, relu2, linear1]),
        (1, 16, 16),
    )


def toy_pad_net() -> nn.Sequential:
    """A toy network containing two padding layers

    Returns:
        nn.Sequential: The corresponding network
    """

    # Input: 1x16x16
    pad_1 = Pad((1, 1, 2, 2), value=1)  # 1x20x18
    conv_1 = nn.Conv2d(1, 3, (2, 2), stride=2)  # 3x10x9
    pad_2 = Pad((1, 1), value=2)  # 3x10x11
    flatten_1 = nn.Flatten()
    lin_1 = nn.Linear(330, 10)

    return nn.Sequential(*[pad_1, conv_1, pad_2, flatten_1, lin_1])


def toy_reshape_net() -> nn.Sequential:
    """A toy network containing three reshape layers

    Returns:
        nn.Sequential: The corresponding network
    """

    class ReshapeLayer(nn.Module):
        def __init__(
            self,
            new_shape: Tuple[int, ...],
        ) -> None:
            super().__init__()
            self.new_shape = new_shape

        def forward(self, x: Tensor) -> Tensor:
            return x.reshape(self.new_shape)

    # Input: 1x16x16
    res_1 = ReshapeLayer(new_shape=(1, 1, 16, 16))  # 1x16x16
    conv_1 = nn.Conv2d(1, 3, (2, 2), stride=2)  # 3x8x8
    res_2 = ReshapeLayer(new_shape=(1, 3, 64))  # 3x64
    res_3 = ReshapeLayer(new_shape=(1, 192))  # 1x192 - Flatten
    lin_1 = nn.Linear(192, 10)

    return nn.Sequential(*[res_1, conv_1, res_2, res_3, lin_1])


def abs_toy_pad_net() -> Tuple[AbstractNetwork, Tuple[int, int, int]]:
    """
    A toy network containing two padding layers

    Returns:
        abstract_network: AS wrapper around the network
        input_size: size of the input
    """
    input_size = (1, 4, 4)

    # Input: 1x16x16
    layers: List[nn.Module] = []
    layers += [Pad((1, 1, 2, 2), value=1)]  # 1x7x7
    layers += [nn.Conv2d(1, 2, (3, 3), stride=2)]  # 3x10x9
    layers += [Pad((1, 1))]  # 3x10x11
    layers += [nn.Flatten()]
    return (
        AbstractNetwork.from_concrete_module(nn.Sequential(*layers), input_size),
        input_size,
    )


def abs_toy_pad_tiny_net() -> Tuple[AbstractNetwork, Tuple[int, int, int]]:
    """
    A toy network containing two padding layers

    Returns:
        abstract_network: AS wrapper around the network
        input_size: size of the input
    """

    input_size = (1, 1, 1)

    layers: List[nn.Module] = []
    layers += [Pad((1, 1, 0, 0), value=1)]  # 1x3x1
    layers += [nn.Flatten()]
    return (
        AbstractNetwork.from_concrete_module(nn.Sequential(*layers), input_size),
        input_size,
    )


def toy_permute_net() -> nn.Sequential:
    """A toy network containing two padding layers"""

    class PermuteLayer(nn.Module):
        def __init__(
            self,
            perm_ind: Tuple[int, ...],
        ) -> None:
            super().__init__()
            self.perm_ind = perm_ind

        def forward(self, x: Tensor) -> Tensor:
            return torch.permute(x, self.perm_ind)

    # Input: 1x4x8
    pad_1 = PermuteLayer((0, 1, 3, 2))  # 1x8x4
    conv_1 = nn.Conv2d(1, 3, (2, 2), stride=2)  # 3x4x2
    pad_2 = PermuteLayer((0, 1, 3, 2))  # 3x2x4
    flatten_1 = nn.Flatten()
    lin_1 = nn.Linear(24, 10)

    return nn.Sequential(*[pad_1, conv_1, pad_2, flatten_1, lin_1])


def toy_unbinary_net() -> AbstractNetwork:
    """A toy network containing all unbinary layers

    Returns:
        abstract_network: AS wrapper around the network
    """

    # Input: 1x4x4
    op_1 = UnbinaryOp(op="add", const_val=torch.Tensor([5]), apply_right=False)  # 1x4x4
    conv_1 = nn.Conv2d(1, 3, (1, 1), stride=1)  # 3x4x4
    op_2 = UnbinaryOp(op="sub", const_val=torch.Tensor([5]), apply_right=False)
    op_3 = UnbinaryOp(
        op="sub", const_val=torch.Tensor([5, 4, 3]).reshape((3, 1, 1)), apply_right=True
    )
    op_4 = UnbinaryOp(op="mul", const_val=torch.Tensor([5]), apply_right=False)
    op_5 = UnbinaryOp(op="div", const_val=torch.Tensor([5]), apply_right=False)
    flatten_1 = nn.Flatten()
    lin_1 = nn.Linear(48, 10)
    return AbstractNetwork.from_concrete_module(
        nn.Sequential(*[op_1, conv_1, op_2, op_3, op_4, op_5, flatten_1, lin_1]),
        (1, 4, 4),
    )


def toy_stack_seq_net() -> AbstractNetwork:
    """A toy network containing stacked sequential layers

    Returns:
        abstract_network: AS wrapper around the network
    """

    linear1 = nn.Linear(2, 2)
    relu1 = nn.ReLU()

    linear2 = nn.Linear(2, 2)
    relu2 = nn.ReLU()

    linear3 = nn.Linear(2, 2)
    relu3 = nn.ReLU()

    linear4 = nn.Linear(2, 2)
    relu4 = nn.ReLU()

    linear_out = nn.Linear(2, 1)

    # (weights duplicated from toy_net)

    linear1.weight.data = torch.tensor([[1.0, 1.0], [1.0, -1.0]])
    linear1.bias.data = torch.zeros(2)

    linear2.weight.data = torch.tensor([[1.0, 1.0], [1.0, -1.0]])
    linear2.bias.data = torch.zeros(2)

    linear3.weight.data = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
    linear3.bias.data = torch.tensor([1.0, 0.0])

    linear4.weight.data = torch.tensor([[1.0, 1.0], [1.0, -1.0]])
    linear4.bias.data = torch.zeros(2)

    linear_out.weight.data = torch.tensor([[1.0, -1.0]])
    linear_out.bias.data = torch.zeros(1)

    return AbstractNetwork.from_concrete_module(
        nn.Sequential(
            nn.Sequential(linear1, relu1),
            nn.Sequential(
                linear2,
                relu2,
                nn.Sequential(
                    linear3, relu3
                ),  # This layer creates full_back_prop=True and has a propagate_call_back
            ),
            nn.Sequential(linear4, relu4, linear_out),
        ),
        input_dim=(2,),
    )


def get_mnist_net() -> Tuple[AbstractNetwork, Tuple[int, ...]]:
    network_path = "networks/mnist_2_50_flattened.pyt"
    original_network = mnist_a_b(2, 50)
    state_dict = torch.load(network_path)
    original_network.load_state_dict(state_dict)
    # original_network = original_network[:-1]
    network = AbstractNetwork.from_concrete_module(original_network, (784,))
    return (network, (784,))


def get_relu_layer() -> Tuple[AbstractNetwork, Tuple[int, ...]]:
    layers = [nn.ReLU()]
    original_network = nn.Sequential(*layers)
    network = AbstractNetwork.from_concrete_module(original_network, (10,))
    return network, (10,)


def get_relu_lin_layer() -> Tuple[AbstractNetwork, Tuple[int, ...]]:
    # layers = [nn.Linear(20, 10), nn.ReLU(), nn.Linear(10, 10, False)]
    # layers[-1].weight.data = torch.eye(10)
    layers = [nn.Linear(20, 10), nn.ReLU()]
    original_network = nn.Sequential(*layers)
    network = AbstractNetwork.from_concrete_module(original_network, (20,))
    return network, (20,)


def get_two_relu_lin_layer() -> Tuple[AbstractNetwork, Tuple[int, ...]]:
    layers = [nn.Linear(20, 20), nn.ReLU()]
    # layers += [nn.Linear(20, 10), nn.ReLU(), nn.Linear(10, 10, False)]
    # layers[-1].weight.data = torch.eye(10)
    layers += [nn.Linear(20, 10), nn.ReLU()]
    original_network = nn.Sequential(*layers)
    network = AbstractNetwork.from_concrete_module(original_network, (20,))
    freeze_network(network)
    return network, (20,)


def get_three_relu_lin_layer() -> Tuple[AbstractNetwork, Tuple[int, ...]]:
    layers = [nn.Linear(784, 50), nn.ReLU()]
    layers += [nn.Linear(50, 50), nn.ReLU()]
    # layers += [nn.Linear(50, 10), nn.ReLU(), nn.Linear(10, 10, False)]
    # layers[-1].weight.data = torch.eye(10)
    layers += [nn.Linear(50, 10), nn.ReLU()]
    original_network = nn.Sequential(*layers)
    network = AbstractNetwork.from_concrete_module(original_network, (784,))
    freeze_network(network)
    return network, (784,)


def toy_convtranspose2d_net() -> Tuple[AbstractNetwork, Tuple[int, int, int]]:
    """
    A toy network containing a convtranspose2d layer

    Returns:
        abstract_network: AS wrapper around the network
        input_size: size of the input
    """
    input_size = (1, 4, 4)

    # Input: 1x4x4
    layers: List[nn.Module] = []
    layers += [
        nn.ConvTranspose2d(
            in_channels=1,
            out_channels=2,
            kernel_size=(2, 2),
            stride=(1, 1),
            padding=(0, 0),
            output_padding=(0, 0),
            bias=True,
            dilation=1,
        )
    ]  # 2x5x5
    layers += [nn.Flatten()]
    return (
        AbstractNetwork.from_concrete_module(nn.Sequential(*layers), input_size),
        input_size,
    )


def get_convtranspose2d_conv_net() -> Tuple[AbstractNetwork, Tuple[int, int, int]]:

    input_size = (1, 4, 4)

    # Input: 1x4x4
    layers: List[nn.Module] = []
    layers += [
        nn.ConvTranspose2d(
            in_channels=1,
            out_channels=3,
            kernel_size=(2, 2),
            stride=(1, 1),
            padding=(0, 0),
            output_padding=(0, 0),
            bias=True,
            dilation=1,
        )
    ]  # 2x5x5
    layers += [
        nn.Conv2d(
            in_channels=3,
            out_channels=5,
            kernel_size=(2, 2),
            stride=(1, 1),
            padding=(1, 1),
            bias=True,
            dilation=1,
        )
    ]  # 5x6x6
    layers += [nn.ReLU()]
    layers += [
        nn.ConvTranspose2d(
            in_channels=5,
            out_channels=1,
            kernel_size=(2, 2),
            stride=(2, 2),
            padding=(0, 0),
            output_padding=(0, 0),
            bias=True,
            dilation=1,
        )
    ]  # 1x12x12
    layers += [nn.Flatten()]
    return (
        AbstractNetwork.from_concrete_module(nn.Sequential(*layers), input_size),
        input_size,
    )


def get_toy_split_block() -> Tuple[AbstractNetwork, Tuple[int, ...]]:
    input_size = (1, 3, 4)

    # Input: 1x3x4
    path = nn.Sequential(*[nn.Conv2d(1, 1, 3, 1, 1), nn.ReLU()])
    layers = [
        SplitBlock(
            split=(False, (3, 1), None, -1, True),
            center_path=path,
            inner_reduce=(1, False, False),
            outer_reduce=(1, False, False),
        ),
        nn.Flatten(),
    ]
    return (
        AbstractNetwork.from_concrete_module(nn.Sequential(*layers), input_size),
        input_size,
    )


def get_nn4sys_128d_splitblock() -> Tuple[AbstractNetwork, Tuple[int, ...]]:
    onnx_path = "vnn-comp-2022-sup/benchmark_vnn22/nn4sys2022/model/mscn_128d.onnx"
    o2p_net, in_shape, in_name = load_onnx_model(onnx_path)
    net = o2p_net[0].paths[2]
    # in_shape = (11, 14)
    net = nn.Sequential(net[2])
    in_shape = (3, 7)
    freeze_network(net)
    net.eval()
    abs_net = AbstractNetwork.from_concrete_module(net, in_shape)
    return abs_net, in_shape


def get_nn4sys_128d_block() -> Tuple[AbstractNetwork, Tuple[int, ...]]:
    onnx_path = "vnn-comp-2022-sup/benchmark_vnn22/nn4sys2022/model/mscn_128d.onnx"
    o2p_net, in_shape, in_name = load_onnx_model(onnx_path)
    net = o2p_net[0].paths[2]
    in_shape = (11, 14)
    # net = nn.Sequential(net[2])
    # in_shape = (3, 7)
    freeze_network(net)
    net.eval()
    abs_net = AbstractNetwork.from_concrete_module(net, in_shape)
    return abs_net, in_shape


def get_nn4sys_128d_multipath_block_stacked() -> Tuple[
    AbstractNetwork, Tuple[int, ...]
]:
    onnx_path = "vnn-comp-2022-sup/benchmark_vnn22/nn4sys2022/model/mscn_128d.onnx"
    o2p_net, in_shape, in_name = load_onnx_model(onnx_path)
    net = o2p_net[0]
    in_shape = (11, 14)
    # net = nn.Sequential(net[2])
    # in_shape = (3, 7)
    freeze_network(net)
    net.eval()
    abs_net = AbstractNetwork.from_concrete_module(net, in_shape)
    return abs_net, in_shape


def get_deep_poly_bounds(
    network: AbstractNetwork,
    input_lb: Tensor,
    input_ub: Tensor,
    use_dependence_sets: bool = False,
    use_early_termination: bool = False,
    reset_input_bounds: bool = True,
    recompute_intermediate_bounds: bool = True,
    max_num_query: int = 10000,
) -> Tuple[Tensor, Tensor]:
    device = input_lb.device
    query_coef = get_output_bound_initial_query_coef(
        dim=network.output_dim,
        intermediate_bounds_to_recompute=None,  # get all bounds
        use_dependence_sets=use_dependence_sets,
        batch_size=1,
        device=device,
        dtype=None,  # TODO: should this be something else?
    )
    abstract_shape = MN_BaB_Shape(
        query_id=query_tag(network),
        query_prev_layer=None,
        queries_to_compute=None,
        lb=AffineForm(query_coef),
        ub=AffineForm(query_coef),
        unstable_queries=None,
        subproblem_state=None,
    )
    output_shape = network.backsubstitute_mn_bab_shape(
        make_backsubstitution_config(
            use_dependence_sets=use_dependence_sets,
            use_early_termination=use_early_termination,
            max_num_query=max_num_query,
        ),
        input_lb,
        input_ub,
        query_coef=None,
        abstract_shape=abstract_shape,
        compute_upper_bound=True,
        reset_input_bounds=reset_input_bounds,
        recompute_intermediate_bounds=recompute_intermediate_bounds,
        optimize_intermediate_bounds=False,
    )
    out_lbs, out_ubs = output_shape.concretize(input_lb, input_ub)
    assert out_ubs is not None
    return (out_lbs, out_ubs)


def get_deep_poly_lower_bounds(
    network: AbstractNetwork,
    input_lb: Tensor,
    input_ub: Tensor,
    use_dependence_sets: bool = False,
    use_early_termination: bool = False,
) -> Tuple[Tensor, Optional[Tensor]]:
    device = input_lb.device
    query_coef = get_output_bound_initial_query_coef(
        dim=network.output_dim,
        intermediate_bounds_to_recompute=None,  # get all bounds
        use_dependence_sets=False,
        batch_size=1,
        device=device,
        dtype=None,  # TODO: should this be something else?
    )
    abstract_shape = MN_BaB_Shape(
        query_id=query_tag(network),
        query_prev_layer=None,
        queries_to_compute=None,
        lb=AffineForm(query_coef),
        ub=None,
        unstable_queries=None,
        subproblem_state=None,
    )
    output_shape = network.backsubstitute_mn_bab_shape(
        config=make_backsubstitution_config(
            use_dependence_sets=use_dependence_sets,
            use_early_termination=use_early_termination,
        ),
        input_lb=input_lb,
        input_ub=input_ub,
        query_coef=None,
        abstract_shape=abstract_shape,
        compute_upper_bound=False,
        reset_input_bounds=True,
        recompute_intermediate_bounds=True,
        optimize_intermediate_bounds=False,
    )
    out_lbs, out_ubs = output_shape.concretize(input_lb, input_ub)
    return (out_lbs, out_ubs)


def get_deep_poly_forward_bounds(
    network: AbstractNetwork,
    input_lb: Tensor,
    input_ub: Tensor,
) -> Tuple[Tensor, Tensor]:
    abstract_shape = DeepPoly_f.construct_from_bounds(input_lb, input_ub)
    in_lb, in_ub = abstract_shape.concretize()
    assert torch.isclose(in_lb, input_lb, atol=1e-10, rtol=1e-10).all()
    assert torch.isclose(in_ub, input_ub, atol=1e-10, rtol=1e-10).all()
    output_shape = network.propagate_abstract_element(
        abstract_shape,
    )
    out_lbs, out_ubs = output_shape.concretize()
    assert out_ubs is not None
    return (out_lbs, out_ubs)


def get_zono_bounds(
    network: AbstractNetwork,
    input_lb: Tensor,
    input_ub: Tensor,
) -> Tuple[Tensor, Tensor]:
    abstract_shape = HybridZonotope.construct_from_bounds(
        input_lb, input_ub, domain="zono"
    )
    output_shape = network.propagate_abstract_element(
        abstract_shape,
    )
    out_lbs, out_ubs = output_shape.concretize()
    assert out_ubs is not None
    return (out_lbs, out_ubs)


def get_input_splitting_bounds(
    network: AbstractNetwork, input_lb: Tensor, input_ub: Tensor, domain_splitting: dict
) -> Tuple[Tensor, Tensor]:
    config = MNBabVerifierConfig(Bunch())
    for k, v in domain_splitting.items():
        if hasattr(config.domain_splitting, k):
            setattr(config.domain_splitting, k, v)
    config.outer.adversarial_attack = False
    verifier = MNBaBVerifier(network, input_lb.device, config)

    dim = int(np.random.randint(0, network.output_dim))
    out_lbs = None
    ub = 100.0
    lb = -100.0
    while ub - lb > 1e-3:
        mid = (ub + lb) / 2
        properties_to_verify = [[(dim, -1, mid)]]
        queue, out_lbs_tmp = verifier._verify_with_input_domain_splitting(
            config.domain_splitting,
            input_lb,
            input_ub,
            properties_to_verify,
            20 + time.time(),
        )
        if len(queue) == 0:
            lb = out_lbs_tmp.detach().item() + mid
            out_lbs = torch.zeros_like(out_lbs_tmp) + lb
        else:
            ub = mid
    assert out_lbs is not None
    return out_lbs, torch.ones_like(out_lbs) * torch.inf


def opt_intermediate_bounds(
    network: AbstractNetwork,
    input_lb: Tensor,
    input_ub: Tensor,
    use_prima: bool = False,
    use_milp: bool = False,
) -> MN_BaB_Shape:
    if use_milp:
        milp_model = MILPNetwork.build_model_from_abstract_net(
            (input_lb + input_ub) / 2, input_lb, input_ub, network
        )

        for idx, layer in enumerate(network.layers):
            if isinstance(layer, ReLU):
                layer.optim_input_bounds = milp_model.get_network_bounds_at_layer_multi(
                    layer_tag(network.layers[idx]), True, 100, 300, time.time() + 300
                )

    device = input_lb.device
    query_coef = get_output_bound_initial_query_coef(
        dim=network.output_dim,
        intermediate_bounds_to_recompute=None,  # get all bounds
        use_dependence_sets=False,
        batch_size=1,
        device=device,
        dtype=None,  # TODO: should this be something else?
    )
    abstract_shape = MN_BaB_Shape(
        query_id=query_tag(network),
        query_prev_layer=None,
        queries_to_compute=None,
        lb=AffineForm(query_coef),
        ub=AffineForm(query_coef),
        unstable_queries=None,
        subproblem_state=SubproblemState.create_default(
            split_state=None,
            optimize_prima=use_prima,
            batch_size=1,
            device=device,
            use_params=True,
        ),
        # best_layer_bounds=None, # TODO: it used to pass None here, is this a problem?
    )
    layer_ids_for_which_to_compute_prima_constraints = []
    config = make_backsubstitution_config()
    if use_prima:
        prima_hyperparameters = make_prima_hyperparameters()
        layer_ids_for_which_to_compute_prima_constraints = (
            network.get_activation_layer_ids()
        )
        config = config.with_prima(
            prima_hyperparameters,
            layer_ids_for_which_to_compute_prima_constraints,
        )

    output_shape = network.backsubstitute_mn_bab_shape(
        config=make_backsubstitution_config(),
        input_lb=input_lb,
        input_ub=input_ub,
        query_coef=None,
        abstract_shape=abstract_shape,
        compute_upper_bound=True,
        reset_input_bounds=True,
        recompute_intermediate_bounds=True,
        optimize_intermediate_bounds=True,
    )
    return output_shape


def run_fuzzing_test(
    as_net: AbstractNetwork,
    input: Tensor,
    input_lb: Tensor,
    input_ub: Tensor,
    input_shape: Tuple[int, ...],
    bounding_call: Callable[
        [AbstractNetwork, Tuple[Tensor, Tensor]], Tuple[Tensor, Tensor]
    ],
    use_beta: bool = True,
    use_adv: bool = False,
    seed: int = 42,
) -> None:
    device = input_lb.device

    (lb, ub) = bounding_call(as_net, (input_lb, input_ub))
    print(f"lb: {lb} ub: {ub}")
    if use_beta:
        seed_everything(seed)
        m = Beta(concentration0=0.5, concentration1=0.5)
        eps = (input_ub - input_lb) / 2
        out = as_net(input)
        lb, ub = lb.to(device), ub.to(device)
        for i in range(100):
            shape_check = (256, *input_shape[1:])
            check_x = input_lb + 2 * eps * m.sample(shape_check).to(device)
            out = as_net(check_x)
            assert (lb - 1e-4 <= out).all() and (
                out <= ub + 1e-4
            ).all(), f"Failed with lb violation: {(lb- out).max()} and ub violation: {(out - ub).max()}"

    if use_adv:
        bounds = (lb.to(device), ub.to(device))
        target = torch.argmax(as_net(input)).item()
        _pgd_whitebox(
            as_net,
            input,
            bounds,
            target,
            input_lb,
            input_ub,
            input.device,
            num_steps=200,
        )


def optimize_output_node_bounds_with_prima_crown(
    network: AbstractNetwork,
    output_idx: int,
    input_lb: Tensor,
    input_ub: Tensor,
    optimize_alpha: bool = False,
    optimize_prima: bool = False,
    custom_optimizer_config: Optional[MNBabOptimizerConfig] = None,
) -> Tuple[float, float]:
    config = (
        make_optimizer_config(
            optimize_alpha=optimize_alpha, optimize_prima=optimize_prima
        )
        if custom_optimizer_config is None
        else custom_optimizer_config
    )
    backsubstitution_config = make_backsubstitution_config()
    optimizer = MNBabOptimizer(config, backsubstitution_config)

    print(f"computing lower bound to x_{output_idx}")
    lb_query_coef = torch.zeros(1, 1, *network.output_dim)
    lb_query_coef.data[(0,) * (lb_query_coef.dim() - 1) + (output_idx,)] = 1
    lb_bounded_subproblem, _ = optimizer.bound_root_subproblem(
        input_lb,
        input_ub,
        network,
        lb_query_coef,
        device=input_lb.device,
    )
    lb = lb_bounded_subproblem.lower_bound

    print(f"computing upper bound to x_{output_idx}")
    ub_query_coef = torch.zeros(1, 1, *network.output_dim)
    ub_query_coef.data[(0,) * (lb_query_coef.dim() - 1) + (output_idx,)] = -1
    ub_bounded_subproblem, _ = optimizer.bound_root_subproblem(
        input_lb,
        input_ub,
        network,
        ub_query_coef,
        device=input_lb.device,
    )
    ub = ub_bounded_subproblem.lower_bound
    ub = (-1) * ub
    return lb, ub


def lower_bound_output_node_with_branch_and_bound(
    network: AbstractNetwork,
    output_idx: int,
    input_lb: Tensor,
    input_ub: Tensor,
    batch_sizes: Sequence[int],
    early_stopping_threshold: Optional[float] = None,
    optimize_alpha: bool = False,
    optimize_prima: bool = False,
) -> float:
    config = make_verifier_config(
        optimize_alpha=optimize_alpha,
        optimize_prima=optimize_prima,
        beta_lr=0.1,
        bab_batch_sizes=batch_sizes,
        recompute_intermediate_bounds_after_branching=True,
    )
    optimizer = MNBabOptimizer(config.optimizer, config.backsubstitution)
    bab = BranchAndBound(
        optimizer, config.bab, config.backsubstitution, torch.device("cpu")
    )

    query_coef = torch.zeros(1, 1, *network.output_dim)
    query_coef.data[0, 0, output_idx] = 1

    return bab.bound_minimum_with_branch_and_bound(
        "dummy_id",
        query_coef,
        network,
        input_lb,
        input_ub,
        early_stopping_threshold,
    )[0]


def prima_crown_wrapper_call(
    optimize_alpha: bool, optimize_prima: bool
) -> Callable[[AbstractNetwork, Tuple[Tensor, Tensor]], Tuple[Tensor, Tensor]]:
    def prima_crown_call(
        network: AbstractNetwork, bounds: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tensor]:

        device = bounds[0].device
        out_dim = np.prod(network.output_dim)
        output_lb_with_alpha = torch.full(
            size=(out_dim,),
            fill_value=-1 * float("inf"),
            dtype=torch.float64,
            device=device,
        )
        output_ub_with_alpha = torch.full(
            size=(out_dim,),
            fill_value=1 * float("inf"),
            dtype=torch.float64,
            device=device,
        )

        prior_type = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        for j in range(out_dim):
            (
                prima_crown_alpha_lb,
                prima_crown_alpha_ub,
            ) = optimize_output_node_bounds_with_prima_crown(
                network,
                j,
                bounds[0],
                bounds[1],
                optimize_alpha=optimize_alpha,
                optimize_prima=optimize_prima,
            )
            output_lb_with_alpha[j] = prima_crown_alpha_lb
            output_ub_with_alpha[j] = prima_crown_alpha_ub
        torch.set_default_dtype(prior_type)
        return (output_lb_with_alpha, output_ub_with_alpha)

    return prima_crown_call


def milp_call(
    net: AbstractNetwork, bounds: Tuple[Tensor, Tensor]
) -> Tuple[Tensor, Tensor]:

    # get_deep_poly_bounds(net, bounds[0], bounds[1])

    if len(bounds[0].shape) in [1, 3]:  # Add batch dimesnion
        input_lb = bounds[0].unsqueeze(0)
        input_ub = bounds[1].unsqueeze(0)
    else:
        input_lb = bounds[0]
        input_ub = bounds[1]
    input = (input_lb + input_ub) / 2

    milp_model = MILPNetwork.build_model_from_abstract_net(
        input, bounds[0], bounds[1], net
    )
    return milp_model.get_network_output_bounds()


def dp_call(
    net: AbstractNetwork, bounds: Tuple[Tensor, Tensor]
) -> Tuple[Tensor, Tensor]:
    return get_deep_poly_bounds(net, bounds[0], bounds[1])


def dpf_call(
    net: AbstractNetwork, bounds: Tuple[Tensor, Tensor]
) -> Tuple[Tensor, Tensor]:
    return get_deep_poly_forward_bounds(net, bounds[0], bounds[1])


def zono_call(
    net: AbstractNetwork, bounds: Tuple[Tensor, Tensor]
) -> Tuple[Tensor, Tensor]:
    return get_zono_bounds(net, bounds[0], bounds[1])


def splitting_call(
    net: AbstractNetwork,
    bounds: Tuple[Tensor, Tensor],
    domain_splitting: dict,
) -> Tuple[Tensor, Tensor]:
    return get_input_splitting_bounds(net, bounds[0], bounds[1], domain_splitting)


def _pgd_whitebox(
    model: AbstractNetwork,
    X: Tensor,
    bounds: Tuple[Tensor, Tensor],
    target: float,
    specLB: Tensor,
    specUB: Tensor,
    device: torch.device,
    num_steps: int = 2000,
    step_size: float = 0.2,
    restarts: int = 1,
    seed: int = 42,
    mode: str = "soundness",
) -> None:
    n_class: int = model(X).shape[-1]
    repeats = int(np.floor(100 / 2 / n_class))
    batch_size = int(repeats * n_class * 2)
    device = X.device
    dtype = X.dtype
    assert mode in ["soundness", "accuracy"]
    D = 1e-5

    seed_everything(seed)

    for _ in range(restarts):
        X_pgd = torch.autograd.Variable(
            X.data.repeat((batch_size,) + (1,) * (X.dim() - 1)), requires_grad=True
        ).to(device)
        random_noise = torch.ones_like(X_pgd).uniform_(-0.5, 0.5) * (specUB - specLB)
        X_pgd = torch.autograd.Variable(
            torch.clamp(X_pgd.data + random_noise, specLB, specUB),
            requires_grad=True,
        )

        lr_scale = specUB - specLB
        pbar = tqdm.trange(num_steps + 1)
        for i in pbar:
            opt = torch.optim.SGD([X_pgd], lr=1e-1)
            opt.zero_grad()
            assert (X_pgd <= specUB).all() and (
                X_pgd >= specLB
            ).all(), "Adv example invalid"

            with torch.enable_grad():
                out = model(X_pgd)

                sub_mat = -1 * torch.eye(n_class, dtype=out.dtype, device=out.device)
                sub_mat[:, target] = 1
                sub_mat[target, :] = 0
                deltas = torch.matmul(out, sub_mat.T)

                if mode == "soundness":
                    if not (
                        (bounds[0] <= out + D).all() and (bounds[1] >= out - D).all()
                    ):
                        violating_index = (
                            (bounds[0] > out.detach())
                            .__or__(bounds[1] < out.detach())
                            .sum(1)
                            .nonzero()[0][0]
                        )
                        print("Violating sample: ", X_pgd[violating_index])
                        print("Corresponding output: ", out[violating_index])
                        print("LB: ", bounds[0])
                        print("UB: ", bounds[1])
                    assert (bounds[0] <= out + D).all() and (
                        bounds[1] >= out - D
                    ).all(), f"max lb violation: {torch.max(bounds[0] - out)}, max ub violation {torch.max(out - bounds[1])}"
                elif mode == "accuracy":
                    if not out.argmax(1).eq(target).all():
                        violating_index = (~out.argmax(1).eq(target)).nonzero()[0][0]
                        assert False, f"Violation: {X_pgd[violating_index]}"

                if mode == "soundness":
                    loss = (
                        torch.cat(
                            [
                                torch.ones(
                                    repeats * n_class, dtype=dtype, device=device
                                ),
                                -torch.ones(
                                    repeats * n_class, dtype=dtype, device=device
                                ),
                            ],
                            0,
                        )
                        * out[
                            torch.eye(n_class, dtype=torch.bool, device=device).repeat(
                                2 * repeats, 1
                            )
                        ]
                    )
                elif mode == "accuracy":
                    loss = (
                        -torch.ones(repeats * n_class * 2, dtype=dtype, device=device)
                        * deltas[
                            torch.eye(n_class, dtype=torch.bool, device=device).repeat(
                                2 * repeats, 1
                            )
                        ]
                    )

            loss.sum().backward()
            pbar.set_description(f"Loss: {loss.sum():.3f}")
            eta = lr_scale * step_size * X_pgd.grad.data.sign()
            X_pgd = torch.autograd.Variable(
                torch.clamp(X_pgd.data + eta, specLB, specUB),
                requires_grad=True,
            )


def set_torch_precision(dtype: torch.dtype) -> Callable[[Callable], Callable]:
    def set_torch_precision_dec(func: Callable) -> Callable[[Any], Any]:
        @functools.wraps(func)
        def wrapper_decorator(*args: Any, **kwargs: Any) -> Any:
            prior_torch_dtype = torch.get_default_dtype()
            torch.set_default_dtype(dtype)
            value = func(*args, **kwargs)
            torch.set_default_dtype(prior_torch_dtype)
            return value

        return wrapper_decorator

    return set_torch_precision_dec
