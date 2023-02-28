import copy
import gzip
import os
import re
import typing
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt


@typing.no_type_check
def identify_var(var_string: str) -> Tuple[str, int]:
    match = re.match(r"([\-,\+]*)([A-Z,a-z,_]+)_([0-9]*)", var_string)
    if match is None:
        assert False, var_string
    else:
        var_group = match.group(2)
        var_idx = int(match.group(3))
    return var_group, var_idx


@typing.no_type_check
def check_numeric(var_string: str, dtype=np.float32) -> float:
    match = re.match(r"([\-,\+]*)([0-9]*(\.[0-9]*)?(e[\-,\+]?[0-9]+)?)", var_string)
    var_string = "".join((var_string).split())
    if match is None or len(match.group(2)) == 0 or match.group(2) is None:
        return None
    else:
        # sign = -1 if match.group(1)=="-" else 1
        try:
            num = dtype(var_string)
            return num
        except Exception:
            assert False, f"Could not translate numeric string {var_string}"


@typing.no_type_check
def extract_terms(
    input_term: List[str], dtype=np.float32
) -> List[Tuple[str, int, float]]:
    terms = [term.strip() for term in input_term.split(" ")]
    sign_flag = None
    output_terms = []
    for term in terms:
        if term == "":
            continue
        if term == "-":
            sign_flag = -1
        elif term == "+":
            sign_flag = +1
        else:
            num = check_numeric(term, dtype)
            if num is None:
                if term.startswith("-"):
                    sign_flag = -1 if sign_flag is None else -1 * sign_flag
                elif term.startswith("+"):
                    sign_flag = +1
                var_group, var_idx = identify_var(term)
                value = 1 if sign_flag is None else sign_flag
            else:
                var_group = "const"
                var_idx = -1
                value = num
            value_f = dtype(value)
            assert isinstance(value_f, dtype)
            output_terms.append((var_group, var_idx, value_f))
            sign_flag = None
    return output_terms


@typing.no_type_check
def identify_variables(
    lines: List[str],
) -> Tuple[List[Tuple[str, int, str]], List[Tuple[str, int, str]]]:
    net_inputs = []
    net_outputs = []

    for line in lines:
        if line.startswith(";"):
            continue
        if line.startswith("(declare-const"):
            match = re.match(
                r"\(declare-const ([A-Z,a-z,_]+)_([0-9]*) ([A-Z,a-z]*)\)", line
            )
            if match is None:
                assert False, line
            else:
                var_group = match.group(1)
                var_idx = int(match.group(2))
                var_type = match.group(3)
                if var_group == "X":
                    net_inputs.append(("X", var_idx, var_type))
                elif var_group == "Y":
                    net_outputs.append(("Y", var_idx, var_type))
                else:
                    assert False, f"Unrecognized variable:\n{line}"
    return net_inputs, net_outputs


@typing.no_type_check
def vnn_lib_data_loader(
    vnn_lib_spec: str,
    dtype=np.float32,
    index_is_nchw: bool = True,
    output_is_nchw: bool = False,
) -> Tuple[
    List[Tuple[List[Tuple[np.float32, ...]], List[int]]],
    Optional[float],
    Optional[float],
    bool,
    bool,
]:
    if os.path.isdir(vnn_lib_spec):
        vnn_lib_spec_files = [
            os.path.join(vnn_lib_spec, f)
            for f in os.listdir(vnn_lib_spec)
            if os.path.isfile(os.path.join(vnn_lib_spec, f))
        ]
    elif os.path.isfile(vnn_lib_spec):
        vnn_lib_spec_files = [vnn_lib_spec]
    else:
        vnn_lib_spec_files = []

    vnn_lib_spec_files = [f for f in vnn_lib_spec_files if f.endswith(".vnnlib")]

    assert len(vnn_lib_spec_files) > 0, "No .vnnlib file found in indicated location"

    boxes, constraints = [], []
    for f in vnn_lib_spec_files:
        boxes_tmp, constraints_tmp = parse_vnn_lib_prop(f, dtype)
        assert all(
            [np.all(box[0] <= box[1]) for box in boxes_tmp]
        ), "Invalid input spec found"
        assert len(boxes_tmp) == 1
        constraints.append(constraints_tmp[0])
        boxes += boxes_tmp

    mean = None
    std = None
    is_nchw = True
    if len(boxes[0][0]) == 28 * 28:
        input_shape = [1, 28, 28]
    elif len(boxes[0][0]) == 3 * 32 * 32:
        input_shape = [3, 32, 32]
    else:
        input_shape = [len(boxes[0][0])]
    if len(input_shape) == 3 and not (index_is_nchw == output_is_nchw):
        if index_is_nchw:
            boxes = [
                (
                    x.reshape(input_shape).transpose(1, 2, 0).flatten(),
                    y.reshape(input_shape).transpose(1, 2, 0).flatten(),
                )
                for x, y in boxes
            ]
        elif output_is_nchw:
            boxes = [
                (
                    x.reshape(input_shape[1:] + input_shape[:1])
                    .transpose(2, 0, 1)
                    .flatten(),
                    y.reshape(input_shape[1:] + input_shape[:1])
                    .transpose(2, 0, 1)
                    .flatten(),
                )
                for x, y in boxes
            ]
    data_loader_test = zip(boxes, constraints)

    return data_loader_test, mean, std, is_nchw, input_shape


def parse_vnn_lib_prop(
    file_path: str, dtype: npt.DTypeLike = np.float32
) -> Tuple[
    List[Tuple[np.ndarray, np.ndarray]], List[List[List[Tuple[int, int, float]]]]
]:

    if file_path.endswith(".gz"):
        file = gzip.open(file_path)
        lines_raw = file.readlines()
        lines = [line.decode() for line in lines_raw]
    else:
        with open(file_path, "r") as f:
            lines = f.readlines()

    # Get all defined variables
    net_inputs, net_outputs = identify_variables(lines)

    # Input constraints of the form net_inputs >=/<= C [spec_anchors, spec_utility, 1]
    C_lb_list = [-np.ones((len(net_inputs)), dtype) * np.inf]
    C_ub_list = [np.ones((len(net_inputs)), dtype) * np.inf]

    # Output constraints of the form C [net_outputs, 1] >= 0
    C_out_list = [np.zeros((0, len(net_outputs) + 1), dtype)]

    # Dictionaries associating variables with indicies
    idx_dict = {f"{x[0]}_{x[1]}": i for i, x in enumerate(sorted(net_inputs))}
    idx_dict["const_-1"] = -1
    idx_dict_out = {f"{x[0]}_{x[1]}": i for i, x in enumerate(sorted(net_outputs))}
    idx_dict_out["const_-1"] = -1

    # Extract all constraints
    open_brackets_n = 0
    block = []
    for line in lines:
        if line.startswith(";"):
            continue
        if open_brackets_n == 0:
            if line.startswith("(assert"):
                open_brackets_n = 0
                open_brackets_n += line.count("(")
                open_brackets_n -= line.count(")")
                block.append(line.strip())
        else:
            open_brackets_n += line.count("(")
            open_brackets_n -= line.count(")")
            block.append(line.strip())
        if open_brackets_n == 0 and len(block) > 0:
            block_str = " ".join(block)
            match = re.match(r"\(assert(.*)\)$", block_str)
            assert match is not None
            C_lb_list, C_ub_list, C_out_list = parse_assert_block(
                match.group(1).strip(),
                C_lb_list,
                C_ub_list,
                C_out_list,
                idx_dict,
                idx_dict_out,
                dtype,
            )
            block = []

    boxes, GT_constraints = translate_output_constraints(
        C_lb_list, C_ub_list, C_out_list
    )
    return boxes, GT_constraints


@typing.no_type_check
def parse_assert_block(
    block, C_lb_list, C_ub_list, C_out_list, idx_dict, idx_dict_out, dtype=np.float32
):
    match = re.match(r"\((or|and|[>,<,=]+)(.*)\)$", block)
    if match is None:
        assert False, block
    else:
        spec_relation = match.group(1)
        spec_content = match.group(2)
        if spec_relation in ["or", "and"]:
            if spec_relation == "or":
                C_lb_list_new = []
                C_ub_list_new = []
                C_out_list_new = []
            open_brackets_n = 0
            mini_block = []
            for c in spec_content:
                if c == "(":
                    open_brackets_n += 1
                    mini_block.append(c)
                elif open_brackets_n > 0:
                    mini_block.append(c)
                    if c == ")":
                        open_brackets_n -= 1
                        if open_brackets_n == 0:
                            mini_block = "".join(mini_block).strip()
                            if spec_relation == "or":
                                (
                                    C_lb_list_tmp,
                                    C_ub_list_tmp,
                                    C_out_list_tmp,
                                ) = parse_assert_block(
                                    mini_block,
                                    copy.deepcopy(C_lb_list),
                                    copy.deepcopy(C_ub_list),
                                    copy.deepcopy(C_out_list),
                                    idx_dict,
                                    idx_dict_out,
                                    dtype,
                                )
                                C_lb_list_new += C_lb_list_tmp
                                C_ub_list_new += C_ub_list_tmp
                                C_out_list_new += C_out_list_tmp
                            elif spec_relation == "and":
                                C_lb_list, C_ub_list, C_out_list = parse_assert_block(
                                    mini_block,
                                    C_lb_list,
                                    C_ub_list,
                                    C_out_list,
                                    idx_dict,
                                    idx_dict_out,
                                    dtype,
                                )
                            mini_block = []
            assert open_brackets_n == 0
            if spec_relation == "or":
                C_lb_list, C_ub_list, C_out_list = (
                    C_lb_list_new,
                    C_ub_list_new,
                    C_out_list_new,
                )
        else:
            n_y = C_out_list[0].shape[-1] - 1
            var_idx, c_lb, c_ub, c_out = parse_assert_content(
                spec_content.strip(), spec_relation.strip(), n_y, idx_dict_out, dtype
            )
            C_lb_list, C_ub_list, C_out_list = add_constraints(
                var_idx, c_lb, c_ub, c_out, C_lb_list, C_ub_list, C_out_list, idx_dict
            )
    return C_lb_list, C_ub_list, C_out_list


@typing.no_type_check
def parse_assert_content(
    spec_content, spec_relation, n_y, idx_dict_out, dtype=np.float64
):
    c_lb, c_ub, c_out = None, None, None
    if spec_content.startswith("("):
        match = re.match(r"\((.*?)\) .*", spec_content)
        if match is None:
            assert False, spec_content
        else:
            first_term = match.group(1)
    else:
        match = re.match(r"^([A-Z,a-z,_,\-,\+,0-9,\.]*).*?", spec_content)
        if match is None:
            assert False, spec_content
        else:
            first_term = match.group(1)
    # get second term of constraint
    if spec_content.endswith(")"):
        match = re.match(r".*\((.*)\)", spec_content)
        if match is None:
            assert False, spec_content
        else:
            second_term = match.group(1)
    else:
        match = re.match(r".*?([A-Z,a-z,_,\.,\-,\+,0-9]*)$", spec_content)
        if match is None:
            assert False, spec_content
        else:
            second_term = match.group(1)
    assert spec_relation in [">=", "<="]
    g_terms = extract_terms(first_term if spec_relation == ">=" else second_term, dtype)
    l_terms = extract_terms(second_term if spec_relation == ">=" else first_term, dtype)
    # Input Constraints
    if len(g_terms) == 1 and g_terms[0][0] == "X":
        # lower bound on input
        var_idx = f"{g_terms[0][0]}_{g_terms[0][1]}"
        assert (
            len(l_terms) == 1 and l_terms[0][0] == "const"
        ), "only box constraints are supported for the input"
        c_lb = l_terms[0][2]
    elif len(l_terms) == 1 and l_terms[0][0] == "X":
        # upper bound on input
        var_idx = f"{l_terms[0][0]}_{l_terms[0][1]}"
        assert (
            len(g_terms) == 1 and g_terms[0][0] == "const"
        ), "only box constraints are supported for the input"
        c_ub = g_terms[0][2]
    else:
        # Output Constraint
        c_out = np.zeros((1, n_y + 1), dtype)
        var_idx = "Y"
        for term in g_terms:
            var_key = f"{term[0]}_{term[1]}"
            assert var_key in idx_dict_out
            c_out[0, idx_dict_out[var_key]] += term[2]
        for term in l_terms:
            var_key = f"{term[0]}_{term[1]}"
            assert var_key in idx_dict_out
            c_out[0, idx_dict_out[var_key]] -= term[2]
    return var_idx, c_lb, c_ub, c_out


@typing.no_type_check
def add_constraints(
    var_idx, c_lb, c_ub, c_out, C_lb_list, C_ub_list, C_out_list, idx_dict
):
    C_out_list_new = []
    for C_lb, C_ub, C_out in zip(C_lb_list, C_ub_list, C_out_list):
        if c_out is not None:
            C_out = np.concatenate([C_out, c_out], axis=0)
        if c_lb is not None:
            C_lb[idx_dict[var_idx]] = max(C_lb[idx_dict[var_idx]], c_lb)
        if c_ub is not None:
            C_ub[idx_dict[var_idx]] = min(C_ub[idx_dict[var_idx]], c_ub)
        C_out_list_new.append(C_out)
    return C_lb_list, C_ub_list, C_out_list_new


@typing.no_type_check  # noqa: C901 # function too complex
def translate_output_constraints(C_lb_list, C_ub_list, C_out_list):
    # Counterexample definition of the form C [net_outputs, 1] >= 0
    unique_lb = []
    unique_ub = []
    lb_map = []
    ub_map = []

    n_last = 2

    for C_lb in C_lb_list:
        for i, C_lb_ref in enumerate(unique_lb[-n_last:]):
            if np.isclose(C_lb_ref, C_lb).all():
                lb_map.append(max(len(unique_lb) - n_last, 0) + i)
                break
        else:
            lb_map.append(len(unique_lb))
            unique_lb.append(C_lb)

    for C_ub in C_ub_list:
        for i, C_ub_ref in enumerate(unique_ub[-n_last:]):
            if np.isclose(C_ub_ref, C_ub).all():
                ub_map.append(max(len(unique_ub) - n_last, 0) + i)
                break
        else:
            ub_map.append(len(unique_ub))
            unique_ub.append(C_ub)

    spec_map = []
    input_specs = []
    for i_lb, i_ub in zip(lb_map, ub_map):
        for i, spec_ref in enumerate(input_specs[-n_last:]):
            if spec_ref == (i_lb, i_ub):
                spec_map.append(max(len(input_specs) - n_last, 0) + i)
                break
        else:
            spec_map.append(len(input_specs))
            input_specs.append((i_lb, i_ub))

    boxes = [
        (unique_lb[input_specs[i_spec][0]], unique_ub[input_specs[i_spec][1]])
        for i_spec in sorted(list(set(spec_map)))
    ]

    C_out_specs = [[] for _ in range(len(input_specs))]
    for i, spec_idx in enumerate(spec_map):
        C_out_specs[spec_idx].append(C_out_list[i])  # or_list of and_arrays

    GT_specs = []
    for C_out_spec in C_out_specs:
        and_list = []
        for and_array in C_out_spec:
            or_list = []
            for i in range(and_array.shape[0]):
                numeric = and_array[i, -1]
                if numeric != 0:
                    l_label = (and_array[i, 0:-1] < 0).nonzero()[0]
                    g_label = (and_array[i, 0:-1] > 0).nonzero()[0]
                    assert (len(l_label) + len(g_label)) == 1
                    if len(g_label) == 1:
                        or_list.append(
                            (
                                -1,
                                int(g_label),
                                -numeric / np.abs(and_array[i, g_label])[0],
                            )
                        )  # intentional negation
                    elif len(l_label) == 1:
                        or_list.append(
                            (
                                int(l_label),
                                -1,
                                numeric / np.abs(and_array[i, l_label])[0],
                            )
                        )  # intentional negation
                    else:
                        assert False
                else:
                    g_label = (and_array[i, 0:-1] == -1).nonzero()[0]
                    l_label = (and_array[i, 0:-1] == 1).nonzero()[0]
                    if len(g_label) == 0:
                        g_label = [-1]
                    if len(l_label) == 0:
                        l_label = [-1]
                    assert len(l_label) == 1 and len(g_label) == 1
                    or_list.append((g_label[0], l_label[0], 0))
            if or_list not in and_list:
                and_list.append(or_list)
            else:
                print("duplicate constraint detected:", and_list, or_list)
        GT_specs.append(and_list)
    return boxes, GT_specs


@typing.no_type_check
def translate_constraints_to_label(GT_specs):
    labels = []
    for and_list in GT_specs:
        label = None
        for or_list in and_list:
            if len(or_list) > 1:
                label = None
                break
            if label is None:
                label = or_list[0][0]
            elif label != or_list[0][0]:
                label = None
                break
        labels.append(label)
    return labels
