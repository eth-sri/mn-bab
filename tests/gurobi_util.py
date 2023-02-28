"""
This code is mostly adapted from ERAN
# source: https://github.com/mnmueller/eran/blob/all_constraints/tf_verify/ai_milp.py
# commit hash: 4d25107db9db743a008eb63c8fa5a4fe8463b16d

"""
from typing import List, Tuple

import numpy as np
from gurobipy import GRB, LinExpr, Model, Var  # type: ignore[import]
from torch import Tensor

from src.abstract_layers.abstract_conv2d import Conv2d
from src.abstract_layers.abstract_flatten import Flatten
from src.abstract_layers.abstract_linear import Linear
from src.abstract_layers.abstract_network import AbstractNetwork
from src.abstract_layers.abstract_normalization import Normalization
from src.abstract_layers.abstract_relu import ReLU


def create_milp_model(
    network: AbstractNetwork, input_lb: Tensor, input_ub: Tensor
) -> Tuple[Model, List[Var]]:
    model = Model("milp")

    model.setParam("OutputFlag", 0)
    model.setParam(GRB.Param.FeasibilityTol, 1e-5)

    var_list = _encode_inputs(model, input_lb, input_ub)

    prev_layer_start_index = 0
    for i, layer in enumerate(network.layers):
        assert layer.input_bounds  # mypy
        if isinstance(layer, Linear):
            output_bounds = layer.propagate_interval(layer.input_bounds)
            new_start_index = _add_linear_layer_constraints_to(
                model,
                var_list,
                layer,
                output_bounds[0],
                output_bounds[1],
                prev_layer_start_index,
            )
            prev_layer_start_index = new_start_index
        elif isinstance(layer, Conv2d):
            output_bounds = layer.propagate_interval(layer.input_bounds)
            new_start_index = _add_conv_layer_constraints_to(
                model,
                var_list,
                layer,
                output_bounds[0],
                output_bounds[1],
                prev_layer_start_index,
            )
            prev_layer_start_index = new_start_index
        elif isinstance(layer, ReLU):
            new_start_index = _add_relu_layer_constraints_to(
                model, var_list, layer, prev_layer_start_index
            )
            prev_layer_start_index = new_start_index
        elif isinstance(layer, Normalization):
            output_bounds = layer.propagate_interval(layer.input_bounds)
            new_start_index = _add_normalization_layer_constraints_to(
                model,
                var_list,
                layer,
                output_bounds[0],
                output_bounds[1],
                prev_layer_start_index,
            )
            prev_layer_start_index = new_start_index
        elif isinstance(layer, Flatten):
            # no constraints to add
            continue
        else:
            raise NotImplementedError

    return model, var_list


def _encode_inputs(model: Model, input_lb: Tensor, input_ub: Tensor) -> List[Var]:
    var_list = []
    flattenend_input_lb = input_lb.detach().flatten().numpy()
    flattenend_input_ub = input_ub.detach().flatten().numpy()
    for i in range(np.prod(input_lb.shape)):
        var_name = "x" + str(i)
        var_lb = flattenend_input_lb[i]
        var_ub = flattenend_input_ub[i]
        var = model.addVar(vtype=GRB.CONTINUOUS, lb=var_lb, ub=var_ub, name=var_name)
        var_list.append(var)
    return var_list


def _add_linear_layer_constraints_to(
    model: Model,
    var_list: List[Var],
    layer: Linear,
    layer_lb: Tensor,
    layer_ub: Tensor,
    prev_start_var_index: int,
) -> int:
    weights = layer.weight.detach().numpy()
    bias = layer.bias.detach().numpy()
    layer_lb = layer_lb.squeeze(0).detach().numpy()
    layer_ub = layer_ub.squeeze(0).detach().numpy()
    start_var_index = len(var_list)

    n_output_neurons = weights.shape[0]
    for i in range(n_output_neurons):
        var_name = "x" + str(start_var_index + i)
        var = model.addVar(
            vtype=GRB.CONTINUOUS,
            lb=layer_lb[i],
            ub=layer_ub[i],
            name=var_name,
        )
        var_list.append(var)

    for i in range(n_output_neurons):
        n_input_neurons = weights.shape[1]

        expr = LinExpr()
        expr += -1 * var_list[start_var_index + i]
        # matmult constraints
        for k in range(n_input_neurons):
            expr.addTerms(weights[i][k], var_list[prev_start_var_index + k])
        expr.addConstant(bias[i])
        model.addLConstr(expr, GRB.EQUAL, 0)
    return start_var_index


def _add_conv_layer_constraints_to(
    model: Model,
    var_list: List[Var],
    layer: Conv2d,
    layer_lb: Tensor,
    layer_ub: Tensor,
    prev_start_var_index: int,
) -> int:
    filters = layer.weight.data
    assert layer.bias is not None
    biases = layer.bias.data
    filter_size = layer.kernel_size

    num_out_neurons = np.prod(layer.output_dim)
    num_in_neurons = np.prod(
        layer.input_dim
    )  # input_shape[0]*input_shape[1]*input_shape[2]
    # print("filters", filters.shape, filter_size, input_shape, strides, out_shape, pad_top, pad_left)

    flattenend_layer_lb = layer_lb.detach().flatten().numpy()
    flattenend_layer_ub = layer_ub.detach().flatten().numpy()

    start = len(var_list)
    for j in range(num_out_neurons):
        var_name = "x" + str(start + j)
        var = model.addVar(
            vtype=GRB.CONTINUOUS,
            lb=flattenend_layer_lb[j],
            ub=flattenend_layer_ub[j],
            name=var_name,
        )
        var_list.append(var)

    for out_z in range(layer.output_dim[0]):
        for out_x in range(layer.output_dim[1]):
            for out_y in range(layer.output_dim[2]):

                dst_ind = (
                    out_z * layer.output_dim[1] * layer.output_dim[2]
                    + out_x * layer.output_dim[2]
                    + out_y
                )
                expr = LinExpr()
                # print("dst ind ", dst_ind)
                expr += -1 * var_list[start + dst_ind]

                for inp_z in range(layer.input_dim[0]):
                    for x_shift in range(filter_size[0]):
                        for y_shift in range(filter_size[1]):
                            x_val = out_x * layer.stride[0] + x_shift - layer.padding[0]
                            y_val = out_y * layer.stride[1] + y_shift - layer.padding[1]
                            if y_val < 0 or y_val >= layer.input_dim[2]:
                                continue
                            if x_val < 0 or x_val >= layer.input_dim[1]:
                                continue
                            mat_offset = (
                                x_val * layer.input_dim[2]
                                + y_val
                                + inp_z * layer.input_dim[1] * layer.input_dim[2]
                            )
                            if mat_offset >= num_in_neurons:
                                continue
                            src_ind = prev_start_var_index + mat_offset
                            # print("src ind ", mat_offset)
                            # filter_index = x_shift*filter_size[1]*input_shape[0]*out_shape[1] + y_shift*input_shape[0]*out_shape[1] + inp_z*out_shape[1] + out_z
                            expr.addTerms(
                                filters[out_z][inp_z][x_shift][y_shift],
                                var_list[src_ind],
                            )

                expr.addConstant(biases[out_z])

                model.addLConstr(expr, GRB.EQUAL, 0)

    return start


def _add_relu_layer_constraints_to(
    model: Model, var_list: List[Var], layer: ReLU, prev_start_var_index: int
) -> int:
    start_var_index = len(var_list)

    n_output_neurons = np.prod(layer.output_dim)
    relu_counter = start_var_index

    assert layer.input_bounds  # mypy
    layer_input_lb = layer.input_bounds[0].flatten().detach().numpy()
    layer_input_ub = layer.input_bounds[1].flatten().detach().numpy()
    crossing_node_idx = list(np.nonzero(layer_input_lb * layer_input_ub < 0)[0])

    temp_idx = np.ones(n_output_neurons, dtype=bool)
    temp_idx[crossing_node_idx] = False
    relax_encode_idx = np.arange(n_output_neurons)[temp_idx]

    if len(crossing_node_idx) > 0:
        for i, __ in enumerate(crossing_node_idx):
            var_name = "x_bin_" + str(start_var_index + i)
            var_bin = model.addVar(vtype=GRB.BINARY, name=var_name)
            var_list.append(var_bin)
            relu_counter += 1

    # relu output variables
    for i in range(n_output_neurons):
        var_name = "x" + str(relu_counter + i)
        upper_bound = max(0.0, layer_input_ub[i])
        var = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=upper_bound, name=var_name)
        var_list.append(var)

    if len(crossing_node_idx) > 0:
        for i, j in enumerate(crossing_node_idx):
            var_bin = var_list[start_var_index + i]

            if layer_input_ub[j] <= 0:
                expr = var_list[relu_counter + j]
                model.addLConstr(expr, GRB.EQUAL, 0)
            elif layer_input_lb[j] >= 0:
                expr = var_list[relu_counter + j] - var_list[prev_start_var_index + j]
                model.addLConstr(expr, GRB.EQUAL, 0)
            else:
                # y <= x - l(1-a)
                expr = (
                    var_list[relu_counter + j]
                    - var_list[prev_start_var_index + j]
                    - layer_input_lb[j] * var_bin
                )
                model.addLConstr(expr, GRB.LESS_EQUAL, -layer_input_lb[j])

                # y >= x
                expr = var_list[relu_counter + j] - var_list[prev_start_var_index + j]
                model.addLConstr(expr, GRB.GREATER_EQUAL, 0)

                # y <= u . a
                expr = var_list[relu_counter + j] - layer_input_ub[j] * var_bin
                model.addLConstr(expr, GRB.LESS_EQUAL, 0)

                # y >= 0
                expr = var_list[relu_counter + j]
                model.addLConstr(expr, GRB.GREATER_EQUAL, 0)

                # indicator constraint
                model.addGenConstrIndicator(
                    var_bin,
                    True,
                    var_list[prev_start_var_index + j],
                    GRB.GREATER_EQUAL,
                    0.0,
                )

    if len(relax_encode_idx) > 0:
        for j in relax_encode_idx:
            if layer_input_ub[j] <= 0:
                expr = var_list[relu_counter + j]
                model.addLConstr(expr, GRB.EQUAL, 0)
            elif layer_input_lb[j] >= 0:
                expr = var_list[relu_counter + j] - var_list[prev_start_var_index + j]
                model.addLConstr(expr, GRB.EQUAL, 0)

    return relu_counter


def _add_normalization_layer_constraints_to(
    model: Model,
    var_list: List[Var],
    layer: Normalization,
    layer_lb: Tensor,
    layer_ub: Tensor,
    prev_start_var_index: int,
) -> int:
    num_out_neurons = int(np.prod(layer.output_dim))

    flattenend_layer_lb = layer_lb.detach().flatten().numpy()
    flattenend_layer_ub = layer_ub.detach().flatten().numpy()

    start = len(var_list)
    for j in range(num_out_neurons):
        var_name = "x" + str(start + j)
        var = model.addVar(
            vtype=GRB.CONTINUOUS,
            lb=flattenend_layer_lb[j],
            ub=flattenend_layer_ub[j],
            name=var_name,
        )
        var_list.append(var)

    means = layer.means.flatten()
    stds = layer.stds.flatten()

    for j in range(num_out_neurons):
        node_index_in_original_layer_shape = np.unravel_index(j, layer.output_dim)
        channel_index = int(node_index_in_original_layer_shape[0])

        expr = LinExpr()
        expr += (var_list[prev_start_var_index + j] - means[channel_index]) / stds[
            channel_index
        ] - 1 * var_list[start + j]
        model.addLConstr(expr, GRB.EQUAL, 0)

    return start
