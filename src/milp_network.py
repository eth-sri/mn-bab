"""
Adapted from https://gitlab.inf.ethz.ch/OU-VECHEV/PARC/-/blob/MILP_encoding/MILP_Encoding/milp_utility.py
9a3a68a6bd86af27755dbf7a38595395a40baae1
"""

from __future__ import annotations

import multiprocessing
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
from gurobipy import GRB, LinExpr, Model  # type: ignore[import]
from torch import (  # TODO: many Tensor type annotations in this module seem to actually be np.ndarray
    Tensor,
)
from tqdm.contrib.concurrent import process_map  # type: ignore[import]

from src.abstract_layers.abstract_avg_pool2d import AvgPool2d
from src.abstract_layers.abstract_bn2d import BatchNorm2d
from src.abstract_layers.abstract_conv2d import Conv2d
from src.abstract_layers.abstract_flatten import Flatten
from src.abstract_layers.abstract_identity import Identity
from src.abstract_layers.abstract_linear import Linear
from src.abstract_layers.abstract_max_pool2d import MaxPool2d
from src.abstract_layers.abstract_network import AbstractNetwork
from src.abstract_layers.abstract_normalization import Normalization
from src.abstract_layers.abstract_pad import Pad
from src.abstract_layers.abstract_permute import Permute
from src.abstract_layers.abstract_relu import ReLU
from src.abstract_layers.abstract_reshape import Reshape
from src.abstract_layers.abstract_residual_block import ResidualBlock
from src.abstract_layers.abstract_sequential import Sequential
from src.abstract_layers.abstract_unbinary_op import UnbinaryOp
from src.state.tags import LayerTag, layer_tag


class Cache:
    model: Any = None
    output_counter: Optional[int] = None
    lbi: Optional[Tensor] = None
    ubi: Optional[Tensor] = None
    time_limit: Optional[float] = None
    terminate_time: Optional[float] = None


class MILPNetwork:
    """
    Represents a MILP-Encoded (part of a) network

    Attributes:
    """

    def __init__(
        self: MILPNetwork,
        model: Any,
        neuron_vars: Dict[str, List[Any]],
        net: AbstractNetwork,
        layer_id_to_prefix_map: Dict[LayerTag, str],
        prefix_to_layer_bounds: Dict[str, Tuple[Tensor, Tensor]],
        previous_layer_map: Dict[str, str],
        input_shape: Any,
    ) -> None:
        self.model = model
        self.neuron_vars = neuron_vars
        self.net = net
        self.layer_id_to_prefix_map = layer_id_to_prefix_map
        self.prefix_to_layer_bounds = prefix_to_layer_bounds
        self.previous_layer_map = previous_layer_map
        self.input_shape = input_shape

    @classmethod
    @torch.no_grad()
    def build_model_from_abstract_net(
        cls: Type[MILPNetwork],
        x: torch.Tensor,
        lb: torch.Tensor,
        ub: torch.Tensor,
        net: AbstractNetwork,
        max_milp_neurons: int = -1,
        feasibility_tol: float = 1e-6,
        up_to_layer_id: Optional[LayerTag] = None,
    ) -> MILPNetwork:

        # Check for intermediate constraints
        output_lb, output_ub = net.set_layer_bounds_via_interval_propagation(
            lb, ub, use_existing_bounds=True
        )

        assert cls._check_layer_bounds(net.layers)

        model = Model("milp")
        model.setParam("OutputFlag", 0)
        model.setParam(GRB.Param.FeasibilityTol, feasibility_tol)

        curr = x
        curr_flattened = curr.flatten()
        n_inputs = curr_flattened.numel()

        numpy_lb: np.ndarray = lb.detach().reshape((1, -1)).cpu().numpy()
        numpy_ub: np.ndarray = ub.detach().reshape((1, -1)).cpu().numpy()

        # Encode Input
        neuron_vars: Dict[str, List[Any]] = {}
        layer_idx = "input"
        neuron_vars[layer_idx] = []
        for j in range(n_inputs):
            neuron_vars[layer_idx] += [
                model.addVar(
                    lb=numpy_lb[0, j],
                    ub=numpy_ub[0, j],
                    vtype=GRB.CONTINUOUS,
                    name=f"input_{j}",
                )
            ]
            neuron_vars[layer_idx][-1].setAttr("Start", curr_flattened[j])

        # Encode Network
        pr_lb, pr_ub = numpy_lb, numpy_ub
        layer_idx_prefix = ""

        layer_id_to_prefix_map: Dict[LayerTag, str] = {}
        prefix_to_layer_bounds: Dict[str, Tuple[Tensor, Tensor]] = {}
        previous_layer_map: Dict[str, str] = {}

        cls._add_layer_list_with_prefix_to_model(
            layer_idx_prefix,
            net.layers,
            model,
            x,
            neuron_vars,
            max_milp_neurons,
            pr_lb,
            pr_ub,
            layer_id_to_prefix_map,
            prefix_to_layer_bounds,
            previous_layer_map,
            layer_idx,
            up_to_layer_id,
        )

        return cls(
            model,
            neuron_vars,
            net,
            layer_id_to_prefix_map,
            prefix_to_layer_bounds,
            previous_layer_map,
            x.shape,
        )

    @classmethod
    def _add_layer_list_with_prefix_to_model(
        cls: Type[MILPNetwork],
        layer_idx_prefix: str,
        layers: nn.ModuleList,
        model: Any,
        x: Tensor,
        neuron_vars: Dict[str, List[Any]],
        max_milp_neurons: int,
        pr_lb: np.ndarray,
        pr_ub: np.ndarray,
        layer_id_to_prefix_map: Dict[LayerTag, str],
        prefix_to_layer_bounds: Dict[str, Tuple[Tensor, Tensor]],
        previous_layer_map: Dict[str, str],
        pr_layer_idx: str,
        up_to_layer_id: Optional[LayerTag] = None,
    ) -> Tuple[str, Tensor, np.ndarray, np.ndarray, int, bool]:

        for layer_id, layer in enumerate(layers):
            layer_idx = "%s%d" % (layer_idx_prefix, layer_id)

            layer_id_to_prefix_map[layer_tag(layer)] = layer_idx

            lb = layer.output_bounds[0].detach().flatten().cpu()
            ub = layer.output_bounds[1].detach().flatten().cpu()
            prefix_to_layer_bounds[layer_idx] = (lb, ub)

            # This backprops refinements we made for relu to the output bounds of the layer before
            if (
                pr_layer_idx in prefix_to_layer_bounds
                and layer.input_bounds is not None
            ):
                prefix_to_layer_bounds[pr_layer_idx] = (
                    torch.maximum(
                        prefix_to_layer_bounds[pr_layer_idx][0],
                        layer.input_bounds[0].detach().flatten().cpu(),
                    ),
                    torch.minimum(
                        prefix_to_layer_bounds[pr_layer_idx][1],
                        layer.input_bounds[1].detach().flatten().cpu(),
                    ),
                )
                if layer.optim_input_bounds is not None:
                    prefix_to_layer_bounds[pr_layer_idx] = (
                        torch.maximum(
                            prefix_to_layer_bounds[pr_layer_idx][0],
                            layer.optim_input_bounds[0].detach().flatten().cpu(),
                        ),
                        torch.minimum(
                            prefix_to_layer_bounds[pr_layer_idx][1],
                            layer.optim_input_bounds[1].detach().flatten().cpu(),
                        ),
                    )

            if isinstance(layer, Sequential):
                new_prefix = f"{layer_idx}."
                (
                    layer_idx,
                    x,
                    pr_lb,
                    pr_ub,
                    max_milp_after_layer,
                    break_layer_id_reached,
                ) = cls._add_layer_list_with_prefix_to_model(
                    new_prefix,
                    layer.layers,
                    model,
                    x,
                    neuron_vars,
                    max_milp_neurons,
                    pr_lb,
                    pr_ub,
                    layer_id_to_prefix_map,
                    prefix_to_layer_bounds,
                    previous_layer_map,
                    pr_layer_idx,
                    up_to_layer_id,
                )
            elif isinstance(layer, ResidualBlock):
                (
                    layer_idx,
                    x,
                    pr_lb,
                    pr_ub,
                    max_milp_after_layer,
                    break_layer_id_reached,
                ) = cls._add_residual_layer(
                    layer_idx,
                    layer,
                    model,
                    x,
                    neuron_vars,
                    max_milp_neurons,
                    pr_lb,
                    pr_ub,
                    layer_id_to_prefix_map,
                    prefix_to_layer_bounds,
                    previous_layer_map,
                    pr_layer_idx,
                    up_to_layer_id,
                )
            else:
                input_shape = tuple(x.shape[1:])
                x = layer(x)
                curr_lb, curr_ub = layer.output_bounds
                numpy_curr_lb: np.ndarray = (
                    curr_lb.detach().reshape((1, -1)).cpu().numpy()
                )
                numpy_curr_ub: np.ndarray = (
                    curr_ub.detach().reshape((1, -1)).cpu().numpy()
                )

                assert (curr_lb <= x + 1e-6).all() and (
                    curr_ub >= x - 1e-6
                ).all(), (
                    f"max violation lb: {(x-curr_lb).min()} ub: {(curr_ub-x).max()}"
                )
                pr_lb, pr_ub, max_milp_after_layer = cls._add_layer_to_model(
                    layer,
                    model,
                    x,
                    neuron_vars,
                    pr_lb,
                    pr_ub,
                    numpy_curr_lb,
                    numpy_curr_ub,
                    layer_idx,
                    pr_layer_idx,
                    input_shape,
                    max_milp_neurons,
                )
                assert layer_idx not in previous_layer_map
                previous_layer_map[layer_idx] = pr_layer_idx
                if up_to_layer_id is not None:
                    break_layer_id_reached = layer_tag(layer) == up_to_layer_id
                else:
                    break_layer_id_reached = False

            max_milp_neurons = max_milp_after_layer
            pr_layer_idx = layer_idx

            if break_layer_id_reached:
                break

        return layer_idx, x, pr_lb, pr_ub, max_milp_neurons, break_layer_id_reached

    @classmethod
    def _check_layer_bounds(cls: Type[MILPNetwork], layers: nn.ModuleList) -> bool:
        has_bounds = True
        for layer in layers:
            if isinstance(layer, Sequential):
                has_bounds &= cls._check_layer_bounds(layer.layers)
            elif isinstance(layer, ResidualBlock):
                has_bounds &= cls._check_layer_bounds(layer.path_a.layers)
                has_bounds &= cls._check_layer_bounds(layer.path_b.layers)
            else:
                has_bounds &= layer.output_bounds is not None
            if not has_bounds:
                return False
        return True

    @classmethod
    def _add_layer_to_model(
        cls: Type[MILPNetwork],
        layer: nn.Module,
        model: Any,
        x: Tensor,
        neuron_vars: Dict[str, List[Any]],
        pr_lb: np.ndarray,
        pr_ub: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        layer_idx: str,
        pr_layer_idx: str,
        in_shape: Tuple[int, ...],
        n_milp_neurons: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray, int]:

        out_shape = tuple(x.shape[1:])
        x_flat = x.flatten().detach().cpu().numpy()

        neuron_vars[layer_idx] = []

        milp_neurons_after_layer = cls._translate_layer(
            model,
            neuron_vars,
            layer,
            pr_layer_idx,
            layer_idx,
            lb[0],
            ub[0],
            pr_lb[0],
            pr_ub[0],
            n_milp_neurons,
            in_shape,
            out_shape,
            feasible_activation=x_flat,
        )
        return lb, ub, milp_neurons_after_layer

    @classmethod
    def _translate_layer(  # noqa: C901
        cls: type[MILPNetwork],
        model: Any,
        neuron_vars: Dict[str, List[Any]],
        layer: nn.Module,
        pr_layer_idx: str,
        layer_idx: str,
        lb: np.ndarray,
        ub: np.ndarray,
        pr_lb: np.ndarray,
        pr_ub: np.ndarray,
        n_milp_neurons: int,
        in_shape: Tuple[int, ...],
        out_shape: Tuple[int, ...],
        feasible_activation: Optional[np.ndarray] = None,
    ) -> int:
        if isinstance(layer, Linear):
            weight: np.ndarray = layer.weight.cpu().detach().numpy()
            bias: np.ndarray = (
                layer.bias.cpu().detach().numpy() if layer.bias is not None else None
            )
            cls._handle_affine(
                model,
                neuron_vars,
                layer_idx,
                pr_layer_idx,
                lb,
                ub,
                weight,
                bias,
                in_shape,
                out_shape,
                feasible_activation,
            )

        elif isinstance(layer, ReLU):
            n_milp_neurons = cls._handle_relu(
                model,
                neuron_vars,
                layer_idx,
                pr_layer_idx,
                lb,
                ub,
                pr_lb,
                pr_ub,
                in_shape,
                out_shape,
                n_milp_neurons,
                feasible_activation,
            )

        elif isinstance(layer, Flatten) or isinstance(layer, Reshape):
            neuron_vars[layer_idx] = neuron_vars[pr_layer_idx]

        elif isinstance(layer, Identity):
            neuron_vars[layer_idx] = neuron_vars[pr_layer_idx]

        elif isinstance(layer, Conv2d):
            weight = layer.weight.cpu().detach().numpy()
            bias = layer.bias.cpu().detach().numpy() if layer.bias is not None else None

            filter_size, stride, pad, dilation = (
                layer.kernel_size,
                layer.stride,
                layer.padding,
                layer.dilation,
            )
            assert dilation == (1, 1), "Dilation != 1 not implemented"

            if isinstance(pad, int):  # TODO: mypy thinks this is never true
                pad_top, pad_left = pad, pad
            elif len(pad) >= 2:
                pad_top, pad_left, = (
                    pad[0],
                    pad[1],
                )

            cls._handle_conv(
                model,
                neuron_vars,
                layer_idx,
                pr_layer_idx,
                lb,
                ub,
                weight,
                bias,
                filter_size,
                stride,
                pad_top,
                pad_left,
                in_shape,
                out_shape,
                feasible_activation,
            )

        elif isinstance(layer, Normalization):
            mean, sigma = (
                layer.mean.cpu().detach().numpy(),
                layer.sigma.cpu().detach().numpy(),
            )
            bias = (-mean / sigma).flatten()
            weight = np.diag(1 / sigma.flatten()).reshape(mean.size, mean.size, 1, 1)

            cls._handle_conv(
                model,
                neuron_vars,
                layer_idx,
                pr_layer_idx,
                lb,
                ub,
                weight,
                bias,
                (1, 1),
                (1, 1),
                0,
                0,
                in_shape,
                out_shape,
                feasible_activation,
            )

        elif isinstance(layer, AvgPool2d):
            kernel_size, stride, pad = layer.kernel_size, layer.stride, layer.padding
            if isinstance(pad, int):
                pad_top, pad_left = pad, pad
            elif len(pad) >= 2:
                pad_top, pad_left = pad[0], pad[1]

            cls._handle_avg_pool_2d(
                model,
                neuron_vars,
                layer_idx,
                pr_layer_idx,
                lb,
                ub,
                kernel_size,
                stride,
                pad_top,
                pad_left,
                in_shape,
                out_shape,
                feasible_activation,
            )

        elif isinstance(layer, MaxPool2d):

            kernel_size, stride, pad, dilation = (
                layer.kernel_size,
                layer.stride,
                layer.padding,
                layer.dilation,
            )
            assert dilation == (1, 1), "Dilation != 1 not implemented"
            if isinstance(pad, int):
                pad_top, pad_left = pad, pad
            elif len(pad) >= 2:
                pad_top, pad_left = pad[0], pad[1]
            cls._handle_max_pool_2d(
                model,
                neuron_vars,
                layer_idx,
                pr_layer_idx,
                lb,
                ub,
                pr_lb,
                pr_ub,
                kernel_size,
                stride,
                pad_top,
                pad_left,
                in_shape,
                out_shape,
                n_milp_neurons,
                feasible_activation,
            )

        elif isinstance(layer, BatchNorm2d):
            mult_term, add_term = layer.mult_term, layer.add_term
            cls._handle_bn2d(
                model,
                neuron_vars,
                layer_idx,
                pr_layer_idx,
                lb,
                ub,
                mult_term,
                add_term,
                in_shape,
                out_shape,
                feasible_activation,
            )

        elif isinstance(layer, Pad):
            pad, val = layer.pad, layer.value
            cls._handle_pad(
                model,
                neuron_vars,
                layer_idx,
                pr_layer_idx,
                lb,
                ub,
                pad,
                val,
                in_shape,
                out_shape,
                feasible_activation,
            )

        elif isinstance(layer, Permute):
            permutation = layer.perm_ind
            cls._handle_permute(
                model,
                neuron_vars,
                layer_idx,
                pr_layer_idx,
                lb,
                ub,
                permutation,
                in_shape,
                out_shape,
                feasible_activation,
            )

        elif isinstance(layer, UnbinaryOp):
            op = layer.op
            c_val = layer.const_val
            apply_right = layer.apply_right
            cls._handle_unbinary_op(
                model,
                neuron_vars,
                layer_idx,
                pr_layer_idx,
                lb,
                ub,
                op,
                c_val,
                apply_right,
                in_shape,
                out_shape,
                feasible_activation,
            )

        else:
            print("unknown layer type: ", layer)
            assert False

        return n_milp_neurons

    @classmethod
    def _add_residual_layer(
        cls: Type[MILPNetwork],
        new_prefix: str,
        layer: ResidualBlock,
        model: Any,
        x: Tensor,
        neuron_vars: Dict[str, List[Any]],
        max_milp_neurons: int,
        pr_lb: np.ndarray,
        pr_ub: np.ndarray,
        layer_id_to_prefix_map: Dict[LayerTag, str],
        prefix_to_layer_bounds: Dict[str, Tuple[Tensor, Tensor]],
        previous_layer_map: Dict[str, str],
        pr_layer_idx: str,
        up_to_layer_id: Optional[LayerTag],
    ) -> Tuple[str, Tensor, np.ndarray, np.ndarray, int, bool]:
        new_prefix_a = f"{new_prefix}.path_a."
        new_prefix_b = f"{new_prefix}.path_b."
        merge_prefix = f"{new_prefix}.merge"
        (
            final_layer_a_idx,
            x_a,
            pr_lb_a,
            pr_ub_a,
            milp_neurons_after_path_a,
            break_layer_id_reached,
        ) = cls._add_layer_list_with_prefix_to_model(
            new_prefix_a,
            layer.path_a.layers,
            model,
            x,
            neuron_vars,
            max_milp_neurons,
            pr_lb,
            pr_ub,
            layer_id_to_prefix_map,
            prefix_to_layer_bounds,
            previous_layer_map,
            pr_layer_idx,
            up_to_layer_id,
        )

        if break_layer_id_reached:
            return (
                final_layer_a_idx,
                x_a,
                pr_lb_a,
                pr_ub_a,
                milp_neurons_after_path_a,
                break_layer_id_reached,
            )
        (
            final_layer_b_idx,
            x_b,
            pr_lb_b,
            pr_ub_b,
            milp_neurons_after_path_b,
            break_layer_id_reached,
        ) = cls._add_layer_list_with_prefix_to_model(
            new_prefix_b,
            layer.path_b.layers,
            model,
            x,
            neuron_vars,
            max_milp_neurons,
            pr_lb,
            pr_ub,
            layer_id_to_prefix_map,
            prefix_to_layer_bounds,
            previous_layer_map,
            pr_layer_idx,
            up_to_layer_id,
        )

        if break_layer_id_reached:
            return (
                final_layer_b_idx,
                x_b,
                pr_lb_b,
                pr_ub_b,
                milp_neurons_after_path_b,
                break_layer_id_reached,
            )

        x_m = layer(x)

        x_m_flat = x_m.flatten().cpu().numpy()
        num_out_neurons = x_m_flat.shape[0]

        assert layer.output_bounds is not None

        lb, ub = (
            layer.output_bounds[0].detach().reshape((1, -1)).cpu().numpy(),
            layer.output_bounds[1].detach().reshape((1, -1)).cpu().numpy(),
        )

        lb_t = layer.output_bounds[0].detach().flatten().cpu()
        ub_t = layer.output_bounds[1].detach().flatten().cpu()
        prefix_to_layer_bounds[merge_prefix] = (lb_t, ub_t)

        neuron_vars[merge_prefix] = []
        for j in range(num_out_neurons):
            var_name = f"x_{merge_prefix}_{j}"
            var = model.addVar(
                vtype=GRB.CONTINUOUS, lb=lb[0, j], ub=ub[0, j], name=var_name
            )
            var.start = x_m_flat[j]
            neuron_vars[merge_prefix].append(var)

        for j in range(num_out_neurons):
            expr = LinExpr()
            expr += -1 * neuron_vars[merge_prefix][j]
            expr += neuron_vars[final_layer_a_idx][j]
            expr += neuron_vars[final_layer_b_idx][j]
            expr.addConstant(0)
            model.addLConstr(expr, GRB.EQUAL, 0)

        if max_milp_neurons == -1:
            milp_left = -1
        else:
            milp_spent = (
                2 * max_milp_neurons
                - milp_neurons_after_path_a
                - milp_neurons_after_path_b
            )
            milp_left = max(0, max_milp_neurons - milp_spent)

        return merge_prefix, x_m, lb, ub, milp_left, break_layer_id_reached

    @classmethod
    def _handle_conv(
        cls: Type[MILPNetwork],
        model: Any,
        neuron_vars: Dict[str, List[Any]],
        layer_idx: str,
        pr_layer_idx: str,
        lb: np.ndarray,
        ub: np.ndarray,
        weight: np.ndarray,
        bias: Optional[np.ndarray],
        kernel_size: Tuple[int, ...],
        stride: Tuple[int, ...],
        pad_top: int,
        pad_left: int,
        in_shape: Tuple[int, ...],
        out_shape: Tuple[int, ...],
        feasible_activation: Optional[np.ndarray] = None,
    ) -> None:

        neuron_vars[layer_idx] = []
        out_ch, out_h, out_w = out_shape
        in_ch, in_h, in_w = in_shape
        in_hw = in_h * in_w
        out_hw = out_h * out_w
        in_neurons = in_ch * in_hw
        out_neurons = out_ch * out_hw

        for j in range(out_neurons):
            var_name = f"x_{layer_idx}_{j}"
            var = model.addVar(vtype=GRB.CONTINUOUS, lb=lb[j], ub=ub[j], name=var_name)
            if feasible_activation is not None:
                var.start = feasible_activation[j]
            neuron_vars[layer_idx].append(var)

        for out_z in range(out_ch):
            for out_y in range(out_h):
                for out_x in range(out_w):
                    out_ind = out_z * out_hw + out_y * out_w + out_x

                    expr = LinExpr()
                    expr += -1 * neuron_vars[layer_idx][out_ind]
                    for in_z in range(in_ch):
                        for y_shift in range(kernel_size[0]):
                            for x_shift in range(kernel_size[1]):
                                in_x = out_x * stride[1] + x_shift - pad_left
                                in_y = out_y * stride[0] + y_shift - pad_top
                                if in_y < 0 or in_y >= in_h:
                                    continue
                                if in_x < 0 or in_x >= in_w:
                                    continue
                                in_ind = in_z * in_hw + in_y * in_w + in_x
                                if in_ind >= in_neurons:
                                    continue
                                expr.addTerms(
                                    weight[out_z][in_z][y_shift][x_shift],
                                    neuron_vars[pr_layer_idx][in_ind],
                                )

                    if bias is not None:
                        expr.addConstant(bias[out_z])
                    model.addLConstr(expr, GRB.EQUAL, 0)

    @classmethod
    def _handle_avg_pool_2d(
        cls: Type[MILPNetwork],
        model: Any,
        neuron_vars: Dict[str, List[Any]],
        layer_idx: str,
        pr_layer_idx: str,
        lb: np.ndarray,
        ub: np.ndarray,
        kernel_size: Tuple[int, ...],
        stride: Tuple[int, ...],
        pad_top: int,
        pad_left: int,
        in_shape: Tuple[int, ...],
        out_shape: Tuple[int, ...],
        feasible_activation: Optional[np.ndarray] = None,
    ) -> None:
        # NOTE Only handles padding with zeros
        neuron_vars[layer_idx] = []
        out_ch, out_h, out_w = out_shape
        in_ch, in_h, in_w = in_shape
        in_hw = in_h * in_w
        out_hw = out_h * out_w
        in_neurons = in_ch * in_hw
        out_neurons = out_ch * out_hw
        norm = 1 / (torch.prod(torch.Tensor(kernel_size)).item())

        for j in range(out_neurons):
            var_name = f"x_{layer_idx}_{j}"
            var = model.addVar(vtype=GRB.CONTINUOUS, lb=lb[j], ub=ub[j], name=var_name)
            if feasible_activation is not None:
                var.start = feasible_activation[j]
                assert (
                    lb[j] <= feasible_activation[j] and feasible_activation[j] <= ub[j]
                )
            neuron_vars[layer_idx].append(var)

        for out_z in range(out_ch):
            for out_y in range(out_h):
                for out_x in range(out_w):
                    out_ind = out_z * out_hw + out_y * out_w + out_x

                    expr = LinExpr()
                    expr += -1 * neuron_vars[layer_idx][out_ind]
                    for x_shift in range(kernel_size[0]):
                        for y_shift in range(kernel_size[1]):
                            in_x = out_x * stride[0] + x_shift - pad_top
                            in_y = out_y * stride[1] + y_shift - pad_left
                            if in_y < 0 or in_y >= in_h:
                                continue
                            if in_x < 0 or in_x >= in_w:
                                continue
                            in_ind = out_z * in_hw + in_y * in_w + in_x
                            if in_ind >= in_neurons:
                                continue
                            expr.addTerms(
                                norm,
                                neuron_vars[pr_layer_idx][in_ind],
                            )
                    model.addLConstr(expr, GRB.EQUAL, 0)

    @classmethod
    def _handle_max_pool_2d(
        cls: Type[MILPNetwork],
        model: Any,
        neuron_vars: Dict[str, List[Any]],
        layer_idx: str,
        pr_layer_idx: str,
        lb: np.ndarray,
        ub: np.ndarray,
        lb_prev: np.ndarray,
        ub_prev: np.ndarray,
        kernel_size: Tuple[int, ...],
        stride: Tuple[int, ...],
        pad_top: int,
        pad_left: int,
        in_shape: Tuple[int, ...],
        out_shape: Tuple[int, ...],
        partial_milp_neurons: int = 0,
        feasible_activation: Optional[np.ndarray] = None,
    ) -> None:
        # NOTE Only handles padding with zeros
        neuron_vars[layer_idx] = []
        out_ch, out_h, out_w = out_shape
        in_ch, in_h, in_w = in_shape
        in_hw = in_h * in_w
        out_hw = out_h * out_w
        in_neurons = in_ch * in_hw
        out_neurons = out_ch * out_hw
        kernel_count = (
            kernel_size[0] ** 2 if len(kernel_size) == 1 else np.prod(kernel_size)
        )

        assert (
            partial_milp_neurons <= 0
        ), "MaxPool only supports full (-1) or no (0) MILP encoding."

        binary_vars = []
        # zero_var = model.addVar(
        #     vtype=GRB.CONTINUOUS, lb=0.0, ub=0.0, name=f"zero_{layer_idx}"
        # )

        for j in range(out_neurons):
            var_name = f"x_{layer_idx}_{j}"
            var = model.addVar(vtype=GRB.CONTINUOUS, lb=lb[j], ub=ub[j], name=var_name)
            if feasible_activation is not None:
                var.start = feasible_activation[j]
            neuron_vars[layer_idx].append(var)

        for out_z in range(out_ch):
            for out_y in range(out_h):
                for out_x in range(out_w):
                    has_gt_pad_const = False
                    out_ind = out_z * out_hw + out_y * out_w + out_x
                    pool_map = []
                    for x_shift in range(kernel_size[0]):
                        for y_shift in range(kernel_size[1]):
                            in_x = out_x * stride[0] + x_shift - pad_top
                            in_y = out_y * stride[1] + y_shift - pad_left
                            in_ind = out_z * in_hw + in_y * in_w + in_x

                            if in_y < 0 or in_x < 0 or in_y >= in_h or in_x >= in_w:
                                # has_gt_pad_const = True
                                pass
                            elif in_ind < in_neurons:
                                pool_map.append(in_ind)
                    max_l_var = pool_map[
                        np.argmax(np.array([lb_prev[i] for i in pool_map]))
                    ]
                    max_l = lb_prev[max_l_var]
                    # if has_gt_pad_const:
                    #     max_l = max(0, max_l)
                    not_dominated = [i for i in pool_map if ub_prev[i] >= max_l]

                    # if len(not_dominated) == 0:
                    #     assert has_gt_pad_const == True
                    #     model.addConstr(
                    #         neuron_vars[layer_idx][out_ind]
                    #         - zero_var,
                    #         GRB.EQUAL,
                    #         0,
                    #     )
                    if len(not_dominated) == 1:
                        model.addLConstr(
                            neuron_vars[layer_idx][out_ind]
                            - neuron_vars[pr_layer_idx][not_dominated[0]],
                            GRB.EQUAL,
                            0,
                        )
                    else:
                        add_expr = LinExpr()
                        add_expr += -neuron_vars[layer_idx][out_ind]
                        binary_expr = LinExpr()
                        for i, in_ind in enumerate(not_dominated):
                            # y >= x
                            expr = (
                                neuron_vars[layer_idx][out_ind]
                                - neuron_vars[pr_layer_idx][in_ind]
                            )
                            model.addLConstr(expr, GRB.GREATER_EQUAL, 0)
                            add_expr += neuron_vars[pr_layer_idx][in_ind]

                            if partial_milp_neurons < 0:
                                var_name = f"b_{layer_idx}_{j}_{i}"
                                var_bin = model.addVar(vtype=GRB.BINARY, name=var_name)
                                binary_vars.append(var_bin)

                                # y <= x + (1-a)*(u_{rest}-l)
                                max_u_rest = max(
                                    [ub_prev[i] for i in not_dominated if i != in_ind]
                                )
                                cst = max_u_rest - lb_prev[in_ind]
                                expr = (
                                    neuron_vars[layer_idx][out_ind]
                                    - neuron_vars[pr_layer_idx][in_ind]
                                    + cst * var_bin
                                )
                                model.addLConstr(expr, GRB.LESS_EQUAL, cst)

                                # indicator constraints
                                model.addGenConstrIndicator(
                                    var_bin,
                                    True,
                                    neuron_vars[layer_idx][out_ind]
                                    - neuron_vars[pr_layer_idx][in_ind],
                                    GRB.EQUAL,
                                    0.0,
                                )
                                binary_expr += var_bin

                        if partial_milp_neurons == 0:
                            if has_gt_pad_const:
                                model.addLConstr(
                                    neuron_vars[layer_idx][out_ind],
                                    GRB.GREATER_EQUAL,
                                    0,
                                )
                            model.addLConstr(
                                add_expr,
                                GRB.GREATER_EQUAL,
                                sum([lb_prev[i] for i in not_dominated]) - max_l,
                            )
                        else:
                            # if has_gt_pad_const:
                            #     var_name = f"b_{layer_idx}_{j}_{-1}"
                            #     var_bin = model.addVar(vtype=GRB.BINARY, name=var_name)
                            #     binary_vars.append(var_bin)
                            #     binary_expr += var_bin
                            #     model.addGenConstrIndicator(
                            #         var_bin,
                            #         True,
                            #         neuron_vars[layer_idx][out_ind] - zero_var,
                            #         GRB.EQUAL,
                            #         0.0,
                            #     )
                            #
                            #     max_u_rest = max([ub_prev[i] for i in not_dominated])
                            #     expr = (
                            #         neuron_vars[layer_idx][out_ind]
                            #         - zero_var
                            #         + max_u_rest * var_bin
                            #     )
                            #     model.addConstr(expr, GRB.LESS_EQUAL, max_u_rest)
                            # only one indicator can be true

                            model.addLConstr(binary_expr, GRB.EQUAL, 1)

    @classmethod
    def _handle_affine(
        cls: Type[MILPNetwork],
        model: Any,
        neuron_vars: Dict[str, List[Any]],
        layer_idx: str,
        pr_layer_idx: str,
        lb: np.ndarray,
        ub: np.ndarray,
        weight: np.ndarray,
        bias: Optional[np.ndarray],
        in_shape: Tuple[int, ...],
        out_shape: Tuple[int, ...],
        feasible_activation: Optional[np.ndarray] = None,
    ) -> None:
        neuron_vars[layer_idx] = []

        in_n = in_shape[-1]
        out_n = out_shape[-1]

        # output of matmult
        for j in range(out_n):
            var_name = "n_{}_{}".format(layer_idx, j)
            var = model.addVar(vtype=GRB.CONTINUOUS, lb=lb[j], ub=ub[j], name=var_name)
            if feasible_activation is not None:
                var.start = feasible_activation[j]
            neuron_vars[layer_idx].append(var)

        for j in range(out_n):
            expr = LinExpr()
            expr += -1 * neuron_vars[layer_idx][j]
            # matmult constraints
            for k in range(in_n):
                expr.addTerms(weight[j][k], neuron_vars[pr_layer_idx][k])
            if bias is not None:
                expr.addConstant(bias[j])
            model.addLConstr(expr, GRB.EQUAL, 0)

    @classmethod
    def _handle_relu(
        cls: Type[MILPNetwork],
        model: Any,
        neuron_vars: Dict[str, List[Any]],
        layer_idx: str,
        pr_layer_idx: str,
        lb: np.ndarray,
        ub: np.ndarray,
        lb_prev: np.ndarray,
        ub_prev: np.ndarray,
        in_shape: Tuple[int, ...],
        out_shape: Tuple[int, ...],
        partial_milp_neurons: int = 0,
        feasible_activation: Optional[np.ndarray] = None,
    ) -> int:
        neuron_vars[layer_idx] = []
        num_neurons = np.prod(in_shape)

        # Determine which neurons to encode with MILP and which with LP (triangle)
        cross_over_idx = list(np.nonzero(np.array(lb_prev) * np.array(ub_prev) < 0)[0])
        width = np.array(ub_prev) - np.array(lb_prev)
        cross_over_idx = sorted(cross_over_idx, key=lambda x: -width[x])
        milp_encode_idx = (
            cross_over_idx[:partial_milp_neurons]
            if partial_milp_neurons >= 0
            else cross_over_idx
        )  # cross_over_idx if use_milp else cross_over_idx[:partial_milp_neurons]
        temp_idx = np.ones(lb_prev.size, dtype=bool)  # type: ignore[arg-type] # mypy bug?
        temp_idx[milp_encode_idx] = False
        lp_encode_idx = np.arange(num_neurons)[temp_idx]
        assert len(lp_encode_idx) + len(milp_encode_idx) == num_neurons

        # Add binary variables to model
        binary_vars = []
        if len(milp_encode_idx) > 0:
            binary_guess = (
                None
                if feasible_activation is None
                else (feasible_activation > 0).astype(int)
            )  # Initialize binary variables with a feasible solution
            for i, j in enumerate(milp_encode_idx):
                var_name = f"b_{layer_idx}_{j}"
                var_bin = model.addVar(vtype=GRB.BINARY, name=var_name)
                if binary_guess is not None:
                    var_bin.start = binary_guess[j]
                binary_vars.append(var_bin)

        # Add ReLU output variables
        if feasible_activation is not None:
            feas_act_post = np.maximum(
                feasible_activation, 0.0
            )  # Initialize output variables with a feasible solution
        for j in range(num_neurons):
            var_name = f"x_{layer_idx}_{j}"
            upper_bound = max(0.0, float(ub_prev[j]))
            var = model.addVar(
                vtype=GRB.CONTINUOUS, lb=0.0, ub=upper_bound, name=var_name
            )
            if feasible_activation is not None:
                var.start = feas_act_post[j]
            neuron_vars[layer_idx].append(var)

        # Add MILP encoding
        if len(milp_encode_idx) > 0:
            for i, j in enumerate(milp_encode_idx):
                var_bin = binary_vars[i]
                var_in = neuron_vars[pr_layer_idx][j]
                var_out = neuron_vars[layer_idx][j]

                if ub_prev[j] <= 0:
                    # stabely inactive
                    expr = var_out
                    model.addLConstr(expr, GRB.EQUAL, 0)
                elif lb_prev[j] >= 0:
                    # stabely active
                    expr = var_out - var_in
                    model.addLConstr(expr, GRB.EQUAL, 0)
                else:
                    # y <= x - l(1-a)
                    expr = var_out - var_in - lb_prev[j] * var_bin
                    model.addLConstr(expr, GRB.LESS_EQUAL, -lb_prev[j])

                    # y >= x
                    expr = var_out - var_in
                    model.addLConstr(expr, GRB.GREATER_EQUAL, 0)

                    # y <= u . a
                    expr = var_out - ub_prev[j] * var_bin
                    model.addLConstr(expr, GRB.LESS_EQUAL, 0)

                    # y >= 0
                    # expr = var_out
                    # model.addLConstr(expr, GRB.GREATER_EQUAL, 0)

                    # indicator constraint
                    model.addGenConstrIndicator(
                        var_bin, True, var_in, GRB.GREATER_EQUAL, 0.0
                    )

        # Add LP encoding
        if len(lp_encode_idx) > 0:
            for j in lp_encode_idx:
                var_in = neuron_vars[pr_layer_idx][j]
                var_out = neuron_vars[layer_idx][j]
                if ub_prev[j] <= 0:
                    expr = var_out
                    model.addLConstr(expr, GRB.EQUAL, 0)
                elif lb_prev[j] >= 0:
                    expr = var_out - var_in
                    model.addLConstr(expr, GRB.EQUAL, 0)
                else:
                    # y >= 0 (already encoded in range of output variable)
                    # expr = var_out
                    # model.addLConstr(expr, GRB.GREATER_EQUAL, 0)

                    # y >= x
                    expr = var_out - var_in
                    model.addLConstr(expr, GRB.GREATER_EQUAL, 0)

                    # y <= (x-l) * u/(u-l)
                    expr = (ub_prev[j] - lb_prev[j]) * var_out - (
                        var_in - lb_prev[j]
                    ) * ub_prev[j]
                    model.addLConstr(expr, GRB.LESS_EQUAL, 0)

        if partial_milp_neurons < 0:
            return partial_milp_neurons
        else:
            return max(0, partial_milp_neurons - len(milp_encode_idx))

    @classmethod
    def _handle_bn2d(
        cls: Type[MILPNetwork],
        model: Any,
        neuron_vars: Dict[str, List[Any]],
        layer_idx: str,
        pr_layer_idx: str,
        lb: np.ndarray,
        ub: np.ndarray,
        mult_term: Tensor,
        add_term: Tensor,
        in_shape: Tuple[int, ...],
        out_shape: Tuple[int, ...],
        feasible_activation: Optional[np.ndarray],
    ) -> None:

        # y = x * c + b
        neuron_vars[layer_idx] = []

        out_neurons = np.prod(out_shape)

        # define variables
        for j in range(out_neurons):
            var_name = "x_{}_{}".format(layer_idx, j)
            var = model.addVar(vtype=GRB.CONTINUOUS, lb=lb[j], ub=ub[j], name=var_name)
            if feasible_activation is not None:
                var.start = feasible_activation[j]
            neuron_vars[layer_idx].append(var)

        # Channelwise normalization
        vars_per_channel = np.prod(out_shape[1:])
        for c in range(out_shape[0]):
            for j in range(vars_per_channel):
                expr = LinExpr()
                expr += -1 * neuron_vars[layer_idx][c * vars_per_channel + j]
                expr.addTerms(
                    mult_term[c], neuron_vars[pr_layer_idx][c * vars_per_channel + j]
                )
                expr.addConstant(add_term[c])
                model.addLConstr(expr, GRB.EQUAL, 0)

    @classmethod
    def _handle_pad(
        cls: type[MILPNetwork],
        model: Any,
        neuron_vars: Dict[str, List[Any]],
        layer_idx: str,
        pr_layer_idx: str,
        lb: np.ndarray,
        ub: np.ndarray,
        pad: Tuple[int, ...],
        val: float,
        in_shape: Tuple[int, ...],
        out_shape: Tuple[int, ...],
        feasible_activation: Optional[np.ndarray],
    ) -> None:

        out_neurons = np.prod(out_shape)

        neuron_vars[layer_idx] = [None] * out_neurons

        # define variables
        pad_var = model.addVar(
            vtype=GRB.CONTINUOUS, lb=val, ub=val, name=f"pad_{layer_idx}"
        )

        pad_l, pad_r, pad_t, pad_b = pad
        width = out_shape[-1]
        height = out_shape[-2]
        channels = out_shape[0]
        pad_b = height - pad_b
        pad_r = width - pad_r

        prev_idx = 0
        # Pad out
        for out_z in range(channels):
            for out_y in range(height):
                for out_x in range(width):
                    out_ind = out_z * height * width + out_y * width + out_x
                    if (
                        out_x < pad_l
                        or out_x >= pad_r
                        or out_y < pad_t
                        or out_y >= pad_b
                    ):
                        neuron_vars[layer_idx][out_ind] = pad_var
                    else:
                        neuron_vars[layer_idx][out_ind] = neuron_vars[pr_layer_idx][
                            prev_idx
                        ]
                        prev_idx += 1

    @classmethod
    def _handle_permute(
        cls: type[MILPNetwork],
        model: Any,
        neuron_vars: Dict[str, List[Any]],
        layer_idx: str,
        pr_layer_idx: str,
        lb: np.ndarray,
        ub: np.ndarray,
        permutation: Tuple[int, ...],
        in_shape: Tuple[int, ...],
        out_shape: Tuple[int, ...],
        feasible_activation: Optional[np.ndarray],
    ) -> None:

        neuron_vars[layer_idx] = []

        out_neurons = np.prod(out_shape)

        # define variables
        for j in range(out_neurons):
            var_name = "x_{}_{}".format(layer_idx, j)
            var = model.addVar(vtype=GRB.CONTINUOUS, lb=lb[j], ub=ub[j], name=var_name)
            if feasible_activation is not None:
                var.start = feasible_activation[j]
            neuron_vars[layer_idx].append(var)

        dist = len(permutation) - len(in_shape)
        perm_ind = tuple([i - dist for i in permutation[dist:]])

        indices = torch.arange(out_neurons).view(in_shape)
        indices = torch.permute(indices, perm_ind).flatten()
        # permute variables
        for i in range(out_neurons):
            # Set equality
            expr = LinExpr()
            expr += (
                -1 * neuron_vars[layer_idx][i] + neuron_vars[pr_layer_idx][indices[i]]
            )
            model.addLConstr(expr, GRB.EQUAL, 0)

    @classmethod
    def _handle_unbinary_op(
        cls: type[MILPNetwork],
        model: Any,
        neuron_vars: Dict[str, List[Any]],
        layer_idx: str,
        pr_layer_idx: str,
        lb: np.ndarray,
        ub: np.ndarray,
        op: str,
        c_val: Tensor,
        apply_right: bool,
        in_shape: Tuple[int, ...],
        out_shape: Tuple[int, ...],
        feasible_activation: Optional[np.ndarray],
    ) -> None:

        neuron_vars[layer_idx] = []

        out_neurons = np.prod(out_shape)

        # define variables
        for j in range(out_neurons):
            var_name = "x_{}_{}".format(layer_idx, j)
            var = model.addVar(vtype=GRB.CONTINUOUS, lb=lb[j], ub=ub[j], name=var_name)
            if feasible_activation is not None:
                var.start = feasible_activation[j]
            neuron_vars[layer_idx].append(var)

        if c_val.shape:
            if c_val.dim() > len(in_shape):
                c_val = c_val.view(in_shape)
        c_val_for_out_neurons = c_val.broadcast_to(in_shape).flatten()
        assert len(c_val_for_out_neurons) == out_neurons
        if op == "add":
            for i in range(out_neurons):
                expr = LinExpr()
                expr += (
                    -1 * neuron_vars[layer_idx][i]
                    + neuron_vars[pr_layer_idx][i]
                    + c_val_for_out_neurons[i]
                )
                model.addLConstr(expr, GRB.EQUAL, 0)
        elif op == "sub":
            for i in range(out_neurons):
                expr = LinExpr()
                if apply_right:
                    expr += (
                        -1 * neuron_vars[layer_idx][i]
                        - neuron_vars[pr_layer_idx][i]
                        + c_val_for_out_neurons[i]
                    )
                else:
                    expr += (
                        -1 * neuron_vars[layer_idx][i]
                        + neuron_vars[pr_layer_idx][i]
                        - c_val_for_out_neurons[i]
                    )
                model.addLConstr(expr, GRB.EQUAL, 0)
        elif op == "mul":
            for i in range(out_neurons):
                expr = LinExpr()
                expr += (
                    -1 * neuron_vars[layer_idx][i]
                    + neuron_vars[pr_layer_idx][i] * c_val_for_out_neurons[i]
                )
                model.addLConstr(expr, GRB.EQUAL, 0)
        elif op == "div":
            for i in range(out_neurons):
                expr = LinExpr()
                if apply_right:
                    assert False, "Non-linear division"
                else:
                    expr += (
                        -1 * neuron_vars[layer_idx][i]
                        + neuron_vars[pr_layer_idx][i] / c_val_for_out_neurons[i]
                    )
                model.addLConstr(expr, GRB.EQUAL, 0)
        else:
            assert False, "Unknown type"

    def get_network_bounds_at_layer_multi(
        self,
        layer_idx: LayerTag,
        compute_input_bounds: bool = False,
        timeout_per_instance: Optional[float] = None,
        timeout_total: Optional[float] = None,
        timeout: Optional[float] = None,
        NUMPROCESSES: int = 2,
        refine_only_unstable: bool = False,
    ) -> Tuple[Tensor, Tensor]:

        model = self.model

        if timeout_per_instance:
            model.setParam(GRB.Param.TimeLimit, timeout_per_instance)
        model.setParam(GRB.Param.Threads, 1)

        model.update()
        model.reset()

        prefix = self.layer_id_to_prefix_map[layer_idx]

        if compute_input_bounds:
            prefix = self.previous_layer_map[prefix]

        curr_lbs, curr_ubs = self.prefix_to_layer_bounds[prefix]
        candidate_vars = self.neuron_vars[prefix]

        Cache.model = model.copy()
        Cache.lbi = curr_lbs
        Cache.ubi = curr_ubs
        Cache.time_limit = timeout_per_instance
        # Cache.output_counter = offset
        Cache.terminate_time = (
            None
            if timeout is None
            else (
                timeout - 10 if timeout_total is None else timeout - timeout_total * 0.2
            )
        )

        inputs: List[Tuple[Tuple[int, str], float, float]] = []
        for i, v in enumerate(candidate_vars):
            if not refine_only_unstable or (
                float(curr_lbs[i]) < 0 and float(curr_ubs[i]) > 0
            ):
                inputs.append(((i, v.VarName), float(curr_lbs[i]), float(curr_ubs[i])))

        print("Num of bounds to refine: ", len(inputs))

        inputs = sorted(
            inputs, key=lambda q: q[2] - q[1], reverse=True
        )  # Sort all neurons by their current width
        input_ids = [idx for (idx, _, _) in inputs]

        resl = -1 * torch.inf * torch.ones((len(curr_lbs),))
        resu = torch.inf * torch.ones((len(curr_ubs),))

        solver_result = []

        if refine_only_unstable:
            solver_result = process_map(
                get_neuron_bound_stable_call, input_ids, max_workers=NUMPROCESSES
            )
        else:
            solver_result = process_map(
                get_neuron_bound_call, input_ids, max_workers=NUMPROCESSES
            )

        refined_indices = []

        for ((soll, solu, addtoindices, runtime), ind) in zip(solver_result, input_ids):
            resl[ind[0]] = soll
            resu[ind[0]] = solu

            if soll > solu:
                print(f"unsound {ind[0]}")

            if addtoindices:
                refined_indices.append(ind[0])

        resl = torch.maximum(resl, curr_lbs)
        resu = torch.minimum(resu, curr_ubs)

        return resl, resu

    def _get_network_bounds_at_layer_single(
        self: MILPNetwork, layer_id: LayerTag, timeout: Optional[int] = 400
    ) -> Tuple[Tensor, Tensor]:

        prefix = self.layer_id_to_prefix_map[layer_id]
        num_neurons = len(self.neuron_vars[prefix])

        lbs = torch.zeros((num_neurons,))
        ubs = torch.zeros((num_neurons,))
        prior_lbs, prior_ubs = self.prefix_to_layer_bounds[prefix]
        for i, var in enumerate(self.neuron_vars[prefix]):
            lb, ub = self._get_neuron_bound(var, prior_lbs[i], prior_ubs[i], timeout)
            lbs[i] = lb
            ubs[i] = ub

        return lbs, ubs

    def get_network_output_bounds(
        self: MILPNetwork, timeout: float = 400 + time.time()
    ) -> Tuple[Tensor, Tensor]:
        output_layer_idx = layer_tag(self.net.layers[-1])
        return self.get_network_bounds_at_layer_multi(
            output_layer_idx,
            timeout_per_instance=timeout - time.time(),
            timeout_total=timeout - time.time(),
            timeout=timeout,
        )

    def verify_properties(
        self: MILPNetwork,
        properties: List[List[Tuple[int, int, float]]],
        timeout_per_instance: Optional[float] = None,
        timeout_total: Optional[float] = None,
        start_time: Optional[float] = None,
        NUMPROCESSES: int = 2,
    ) -> Tuple[bool, List[float], Optional[Tensor]]:
        output_layer_idx = layer_tag(self.net.layers[-1])
        if isinstance(self.net.layers[-1], Sequential):
            output_layer_idx = layer_tag(self.net.layers[-1].layers[-1])
        model = self.model

        if timeout_per_instance:
            model.setParam(GRB.Param.TimeLimit, timeout_per_instance)
        model.setParam(GRB.Param.Threads, 2)

        model.update()
        model.reset()

        prefix = self.layer_id_to_prefix_map[output_layer_idx]
        candidate_vars = self.neuron_vars[prefix]

        logit_lbs, logit_ubs = self.prefix_to_layer_bounds[prefix]

        lbs = []
        is_verified = True
        ctr_example: Optional[Tensor] = None
        runtime = 0
        D = 1e-4

        for i, prop in enumerate(properties):
            terminate_time = (
                None
                if start_time is None or timeout_total is None
                else start_time + timeout_total * 0.8
            )
            if terminate_time is not None and terminate_time - time.time() < 1:
                break
            label, other_label, offset = prop[0]
            if label == -1:
                lb = offset - logit_ubs[other_label]
            elif other_label == -1:
                lb = logit_lbs[label] - offset
            else:
                lb = logit_lbs[label] - logit_ubs[other_label] - offset

            model.setParam(
                GRB.Param.TimeLimit,
                max(
                    0,
                    min(
                        np.inf
                        if timeout_per_instance is None
                        else timeout_per_instance,
                        np.inf
                        if terminate_time is None
                        else (terminate_time - time.time()),
                    ),
                ),
            )

            obj: Any = LinExpr() + offset * (-1 if label >= 0 else 1)
            obj += (
                model.getVarByName(candidate_vars[label].varName) if label >= 0 else 0
            )
            obj += (
                -model.getVarByName(candidate_vars[other_label].varName)
                if other_label >= 0
                else 0
            )
            # print (f"{ind} {model.getVarByName(ind[1]).VarName}")

            model.setObjective(obj, GRB.MINIMIZE)
            # model.setParam("BestObjStop", -D)
            # model.setParam("CUTOFF", D)
            model.setParam("Threads", multiprocessing.cpu_count())
            model.reset()
            model.optimize(get_milp_callback_for_target_cutoff(0, maximize=False))
            model.optimize()
            runtime += model.RunTime
            assert model.Status not in [
                3,
                4,
            ], "Infeasible Model encountered in refinement"
            if model.Status == 6:
                print("MILP CUTOFF triggered")
                lb = 0  # model.objbound if hasattr(model, "objbound") else curr_lbs[-1]
            elif model.Status == 15:
                print("MILP BestObjStop triggered")
                lb = model.objbound if hasattr(model, "objbound") else lb
                assert model.SolCount > 0
            else:
                lb = (
                    model.objbound
                    if hasattr(model, "objbound") and model.objbound is not None
                    else lb
                )
            if lb <= 0:
                is_verified = False
                if model.SolCount > 0:
                    print("Found Counterexample")
                    ctr_example = torch.tensor(
                        [v.x for v in model.getVars() if "input" in v.var_name]
                    )

            lbs.append(lb)

            if not is_verified:
                return is_verified, lbs, ctr_example

        return is_verified, lbs, None

    def _get_neuron_bound(
        self: MILPNetwork,
        var: Any,
        prior_lb: Tensor,
        prior_ub: Tensor,
        timeout: Optional[int],
    ) -> Tuple[Tensor, Tensor]:
        model = self.model
        ub = _solve_model_for_objective(model, var, timeout, minimize=False)
        if ub is None:
            ub = prior_ub
        model.reset()
        lb = _solve_model_for_objective(model, var, timeout, minimize=True)
        if lb is None:
            lb = prior_lb
        model.reset()

        return lb, ub

    def clone(self: MILPNetwork, full: bool = True) -> MILPNetwork:
        return MILPNetwork(
            self.model.copy(),
            self.neuron_vars,
            self.net,
            self.layer_id_to_prefix_map,
            self.prefix_to_layer_bounds,
            self.previous_layer_map,
            self.input_shape,
        )


def get_neuron_bound_call(ind: Tuple[int, str]) -> Tuple[Tensor, Tensor, bool, float]:
    # Call solver to compute neuronwise bounds in parallel
    if (
        Cache.lbi is None
        or Cache.ubi is None
        or Cache.model is None
        or Cache.time_limit is None
    ):
        raise RuntimeError("Cache not properly set")
    if Cache.terminate_time is not None and Cache.terminate_time - time.time() < 1:
        return Cache.lbi[ind[0]], Cache.ubi[ind[0]], False, 0

    model = Cache.model.copy()
    runtime = 0

    soll, m_runtime = get_neuron_bound_single(model, ind, maximize=False)
    runtime += m_runtime
    solu, m_runtime = get_neuron_bound_single(model, ind, maximize=True)
    runtime += m_runtime

    soll = max(soll, Cache.lbi[ind[0]])
    solu = min(solu, Cache.ubi[ind[0]])

    addtoindices = (soll > Cache.lbi[ind[0]]) or (solu < Cache.ubi[ind[0]])

    return soll, solu, addtoindices, runtime


def get_neuron_bound_stable_call(
    ind: Tuple[int, str]
) -> Tuple[Tensor, Tensor, bool, float]:
    # Call solver to compute neuronwise bounds in parallel
    if (
        Cache.lbi is None
        or Cache.ubi is None
        or Cache.model is None
        or Cache.time_limit is None
    ):
        raise RuntimeError("Cache not properly set")
    if Cache.terminate_time is not None and Cache.terminate_time - time.time() < 1:
        return Cache.lbi[ind[0]], Cache.ubi[ind[0]], False, 0

    model = Cache.model.copy()
    runtime = 0

    if torch.abs(Cache.lbi[ind[0]]) < torch.abs(Cache.ubi[ind[0]]):

        soll, m_runtime = get_neuron_bound_single(
            model, ind, maximize=False, use_cutoff=True
        )
        runtime += m_runtime

        # Stable
        if soll >= 0:
            return soll, Cache.ubi[ind[0]], True, runtime

        solu, m_runtime = get_neuron_bound_single(
            model, ind, maximize=True, use_cutoff=True
        )
        runtime += m_runtime

    else:

        solu, m_runtime = get_neuron_bound_single(
            model, ind, maximize=True, use_cutoff=True
        )
        runtime += m_runtime

        # Stable
        if solu <= 0:
            return Cache.lbi[ind[0]], solu, True, runtime

        soll, m_runtime = get_neuron_bound_single(
            model, ind, maximize=False, use_cutoff=True
        )
        runtime += m_runtime

    soll = max(soll, Cache.lbi[ind[0]])
    solu = min(solu, Cache.ubi[ind[0]])

    addtoindices = (soll > Cache.lbi[ind[0]]) or (solu < Cache.ubi[ind[0]])

    return soll, solu, addtoindices, runtime


def get_neuron_bound_single(
    model: Any, ind: Tuple[int, str], maximize: bool = False, use_cutoff: bool = False
) -> Tuple[Any, Any]:

    obj: Any = LinExpr()
    obj += model.getVarByName(ind[1])

    assert Cache.time_limit is not None

    model.setParam(
        GRB.Param.TimeLimit,
        max(
            0,
            min(
                Cache.time_limit,
                np.inf
                if Cache.terminate_time is None
                else (Cache.terminate_time - time.time()),
            ),
        ),
    )

    if maximize:
        model.setObjective(obj, GRB.MAXIMIZE)
    else:
        model.setObjective(obj, GRB.MINIMIZE)
    model.reset()
    model._vars = model.getVarByName(ind[1])

    # D = 1e-4
    # model.setParam("BestObjStop", -D)
    # model.setParam("CUTOFF", D)
    # model.optimize()

    if use_cutoff:
        model.optimize(get_milp_callback_for_target_cutoff(0, maximize))
    else:
        model.optimize()

    has_objbound = hasattr(model, "objbound")
    model_bound = 0 if model.Status == 6 else (model.objbound if has_objbound else None)

    assert model.Status not in [3, 4], "Infeasible Model encountered in refinement"

    assert Cache.ubi is not None and Cache.lbi is not None
    if maximize:
        res = (
            min(model_bound, Cache.ubi[ind[0]])
            if model_bound is not None
            else Cache.ubi[ind[0]]
        )
    else:
        res = (
            max(model_bound, Cache.lbi[ind[0]])
            if model_bound is not None
            else Cache.lbi[ind[0]]
        )

    return res, model.RunTime


def solver_query_call(
    query: Tuple[int, Tuple[int, str], Tuple[int, str], float]
) -> Tuple[float, Optional[Tensor], float]:
    # Solve for label_ind - other_ind - offset >= 0
    query_index, label_ind, other_ind, offset = query

    if (
        Cache.lbi is None
        or Cache.ubi is None
        or Cache.model is None
        or Cache.time_limit is None
    ):
        raise RuntimeError("Cache not properly set")
    if Cache.terminate_time is not None and Cache.terminate_time - time.time() < 1:
        return float(Cache.lbi[query_index]), None, 0

    model = Cache.model.copy()
    model.setParam(
        GRB.Param.TimeLimit,
        max(
            0,
            min(
                Cache.time_limit,
                np.inf
                if Cache.terminate_time is None
                else (Cache.terminate_time - time.time()),
            ),
        ),
    )
    runtime = 0

    obj: Any = LinExpr() - offset
    obj += model.getVarByName(label_ind[1]) if label_ind[1] is not None else 0
    obj += -model.getVarByName(other_ind[1]) if other_ind[1] is not None else 0
    # print (f"{ind} {model.getVarByName(ind[1]).VarName}")

    model.setObjective(obj, GRB.MINIMIZE)
    model.reset()
    model.optimize(get_milp_callback_for_target_cutoff(0, maximize=False))
    runtime += model.RunTime
    assert model.Status not in [3, 4], "Infeasible Model encountered in refinement"
    if model.Status == 11:
        print("MILP callback")
        lb = model.objbound if hasattr(model, "objbound") else Cache.lbi[query_index]
    else:
        lb = Cache.lbi[query_index] if model.SolCount == 0 else model.objbound
    ctr_example: Optional[Tensor] = None
    if lb <= 0 and model.SolCount != 0:
        print("Found Counterexample")
        ctr_example = torch.tensor(
            [v.x for v in model.getVars() if "input" in v.var_name]
        )

    if label_ind[0] < 0:
        lb = -lb

    return lb, ctr_example, runtime


def _solve_model_for_objective(
    model: Any, obj: Any, timeout: Optional[int], minimize: bool = True
) -> Any:
    if timeout:
        model.setParam(GRB.Param.TimeLimit, timeout)

    if minimize:
        model.setObjective(obj, GRB.MINIMIZE)
    else:
        model.setObjective(obj, GRB.MAXIMIZE)
    model.optimize()
    if model.status in [3, 4]:
        print("Infeasible model encountered - resorting to box")
        return None
    if hasattr(model, "objVal"):
        return model.objVal
    else:
        return model.objbound


def get_milp_callback_for_target_cutoff(
    target: float,
    maximize: bool = True,
) -> Callable[[Any, Any], None]:
    D = 1e-4

    def milp_callback(model: Any, where: Any) -> None:
        if where == GRB.Callback.MIP:
            sol_count = model.cbGet(GRB.Callback.MIP_SOLCNT)
            if sol_count > 0:
                sol_best = model.cbGet(GRB.Callback.MIP_OBJBST)
            obj_bound = model.cbGet(GRB.Callback.MIP_OBJBND)
            if not maximize and (
                obj_bound >= target + D or (sol_count > 0 and sol_best <= target - D)
            ):
                # print(f"Terminated Min: {obj_bound}")
                model.terminate()
            elif maximize and (
                obj_bound <= target - D or (sol_count > 0 and sol_best >= target + D)
            ):
                # print(f"Terminated Max: {obj_bound}")
                model.terminate()

    return milp_callback
