import argparse
import os.path
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import dill  # type: ignore[import]
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from src.abstract_layers.abstract_network import AbstractNetwork
from src.abstract_layers.abstract_sig_base import SigBase
from src.abstract_layers.abstract_sigmoid import Sigmoid, d_sig, sig
from src.abstract_layers.abstract_tanh import Tanh, d_tanh, tanh
from src.mn_bab_verifier import MNBaBVerifier
from src.state.subproblem_state import SubproblemState
from src.utilities.argument_parsing import get_config_from_json
from src.utilities.attacks import _evaluate_cstr
from src.utilities.config import Config, Dtype, make_config
from src.utilities.initialization import seed_everything
from src.utilities.loading.network import freeze_network, load_onnx_model
from src.utilities.output_property_form import OutputPropertyForm
from src.verification_instance import (
    generate_adv_output_str,
    get_sigmoid_gt_constraint,
    get_unet_gt_constraint,
)

FILE_DIR = os.path.realpath(os.path.dirname(__file__))
TEMP_RUN_DIR = os.path.realpath(os.path.join(FILE_DIR, "..", "run"))


def generate_constraints(
    class_num: int, y: int
) -> List[List[List[Tuple[int, int, float]]]]:
    return [[[(y, i, 0)] for i in range(class_num) if i != y]]


def load_io_constraints(
    metadata: Dict[str, str], torch_dtype: torch.dtype, torch_device: torch.device
) -> Tuple[
    List[Tensor], List[Tuple[Tensor, Tensor]], List[List[List[Tuple[int, int, float]]]]
]:

    input_path = f"{TEMP_RUN_DIR}/inputs"
    inputs: List[Tensor] = []
    stack_input = torch.load(f"{input_path}/inputs.pt")
    inputs = list(torch.split(stack_input, 1, dim=0))
    # inputs = [inp.squeeze(dim=0) for inp in inputs]
    assert all([inp.shape[0]==1 for inp in inputs])

    input_region_path = f"{TEMP_RUN_DIR}/input_regions"
    input_regions: List[Tuple[Tensor, Tensor]] = []
    stack_lbs = torch.load(f"{input_region_path}/input_lbs.pt")
    stack_ubs = torch.load(f"{input_region_path}/input_ubs.pt")
    input_regions = [(lb, ub) for (lb, ub) in zip(stack_lbs, stack_ubs)]

    target_g_t_constraint_path = f"{TEMP_RUN_DIR}/io_constraints"

    with open(f"{target_g_t_constraint_path}/target_g_t_constraints.pkl", "rb") as file:
        target_g_t_constraints: List[List[List[Tuple[int, int, float]]]] = dill.load(
            file
        )

    return inputs, input_regions, target_g_t_constraints


def get_asnet(
    net_path: str, config: Config, device: torch.device
) -> Tuple[nn.Module, AbstractNetwork]:
    # Get bounds
    net_format = net_path.split(".")[-1]
    if net_format in ["onnx", "gz"]:
        net_seq, onnx_shape, inp_name = load_onnx_model(net_path)  # Like this for mypy
        net: nn.Module = net_seq
    else:
        assert False, f"No net loaded for net format: {net_format}."

    net.to(device)
    net.eval()
    freeze_network(net)

    if config.dtype == Dtype.float64:
        net = net.double()
    else:
        net = net.float()

    assert isinstance(net, nn.Sequential)
    as_net = AbstractNetwork.from_concrete_module(net, config.input_dim).to(device)
    freeze_network(as_net)
    return net, as_net


def load_meta_data(metadata_path: str) -> Dict[str, str]:
    with open(metadata_path, "r") as f:
        lines = f.readlines()
    metadata: Dict[str, str] = {}
    for line in lines:
        k, v = line.split(":")[:2]
        k = k.strip()
        v = v.strip()
        metadata[k] = v

    return metadata


def run_instance(
    benchmark: str, net_path: str, spec_path: str, res_path: str, timeout: int
) -> None:

    # net_path = os.path.realpath(os.path.join(FILE_DIR, "../..", "vnncomp2022_benchmarks", net_path))
    # spec_path = os.path.realpath(os.path.join(FILE_DIR, "../..", "vnncomp2022_benchmarks", spec_path))
    # res_path = os.path.realpath(os.path.join(FILE_DIR, "../..", "vnncomp2022_benchmarks", res_path))

    shutil.rmtree(f"{res_path}", ignore_errors=True)

    metadata_path = f"{TEMP_RUN_DIR}/metadata.txt"
    config_path = f"{TEMP_RUN_DIR}/config.json"
    abs_network_path = f"{TEMP_RUN_DIR}/abs_network.onnx"

    # 1. Check metadata (Shallow)
    metadata = load_meta_data(metadata_path)
    assert (
        metadata["benchmark"] == benchmark
    ), f"Benchmarks don't match {metadata['benchmark']} {benchmark}"
    assert (
        metadata["network_path"] == net_path
    ), f"Networks don't match {metadata['network_path']} {net_path}"
    assert (
        metadata["spec_path"] == spec_path
    ), f"Specs don't match {metadata['spec_path']} {spec_path}"

    # Get Config
    config = get_config_from_json(config_path)
    parsed_config = make_config(**config)
    seed_everything(parsed_config.random_seed)
    # Set timeout
    parsed_config.timeout = timeout

    if torch.cuda.is_available() and parsed_config.use_gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if parsed_config.dtype == Dtype.float64:
        torch.set_default_dtype(torch.float64)
        dtype = torch.float64
    else:
        torch.set_default_dtype(torch.float32)
        dtype = torch.float32

    # Get data
    (inputs, input_regions, target_g_t_constraints) = load_io_constraints(
        metadata, torch_dtype=dtype, torch_device=device
    )

    network: AbstractNetwork = torch.load(abs_network_path)
    network.eval()
    network.to(dtype).to(device)
    network.set_activation_layers()

    check_sigmoid_tanh_tangent(network)

    sigmoid_encode_property: bool = False
    if isinstance(network.layers[-1], Sigmoid):
        network.layers = network.layers[:-1]
        network.set_activation_layers()
        sigmoid_encode_property = True

    # Get verifier
    verifier = MNBaBVerifier(network, device, parsed_config.verifier)

    start_time = time.time()
    adv_example = None

    if parsed_config.verifier.outer.instance_pre_filter_batch_size is not None:
        assert (
            "unet" not in parsed_config.benchmark_instances_path
        )  # TODO: should this be supported?
        batch_size = parsed_config.verifier.outer.instance_pre_filter_batch_size
        is_verified, adv_example = verify_properties_with_deep_poly_pre_filter(
            parsed_config,
            network,
            verifier,
            inputs,
            input_regions,
            target_g_t_constraints,
            sigmoid_encode_property,
            device=device,
            batch_size=batch_size,
            timeout=parsed_config.timeout + start_time,
        )
    else:  # TODO: should this be in its own function instead, e.g. in mn_bab_verifier.py?
        for input_point, (input_lb, input_ub), gt_constraint in zip(
            inputs, input_regions, target_g_t_constraints
        ):
            if input_lb.dim() == len(parsed_config.input_dim):  # No batch dimension
                input_lb = input_lb.unsqueeze(0)
                input_ub = input_ub.unsqueeze(0)
                input_point = input_point.unsqueeze(0)
            assert tuple(input_lb.shape[1:]) == parsed_config.input_dim

            if sigmoid_encode_property:
                gt_constraint = get_sigmoid_gt_constraint(gt_constraint)

            if "unet" in parsed_config.benchmark_instances_path:
                gt_constraint, gt_target = get_unet_gt_constraint(
                    input_point, (input_lb, input_ub), gt_constraint
                )

            assert isinstance(network, AbstractNetwork)
            out_prop_form = OutputPropertyForm.create_from_properties(
                properties_to_verify=gt_constraint,
                disjunction_adapter=None,
                use_disj_adapter=parsed_config.verifier.outer.use_disj_adapter,
                n_class=network.output_dim[-1],
                device=device,
                dtype=torch.get_default_dtype(),
            )

            if "unet" in parsed_config.benchmark_instances_path:

                assert gt_target
                (is_verified, _, _, _, _,) = verifier.verify_unet_via_config(
                    0,
                    input_point,
                    input_lb,
                    input_ub,
                    out_prop_form,
                    verification_target=gt_target,
                    timeout=parsed_config.timeout + start_time,
                )

            else:

                if out_prop_form.disjunction_adapter is not None:
                    verifier.append_out_adapter(
                        out_prop_form.disjunction_adapter,
                        device,
                        torch.get_default_dtype(),
                    )

                (
                    is_verified,
                    adv_example,
                    lower_idx,
                    lower_bound_tmp,
                    upper_bound_tmp,
                ) = verifier.verify_via_config(
                    0,
                    input_point,
                    input_lb,
                    input_ub,
                    out_prop_form,
                    timeout=parsed_config.timeout + start_time,
                )

                if out_prop_form.disjunction_adapter is not None:
                    verifier.remove_out_adapter()

                if adv_example is not None:
                    assert not is_verified
                    break
                if not is_verified:
                    break

    total_time = time.time() - start_time

    Path("/".join(res_path.split("/")[:-1])).mkdir(parents=True, exist_ok=True)
    with open(res_path, "w") as f:
        if is_verified:
            f.write("unsat\n")
        elif adv_example is not None:
            f.write("sat\n")
            counter_example = generate_adv_output_str(
                adv_example[0], network, input_regions[0][0].shape, device, dtype
            )
            f.write(counter_example)
        elif total_time >= timeout:
            f.write("timeout\n")
        else:
            f.write("unknown\n")


def verify_properties_with_deep_poly_pre_filter(  # noqa: C901 # TODO: simplify function. should this be in mn_bab_verifier.py instead?
    parsed_config: Config,
    network: AbstractNetwork,
    verifier: MNBaBVerifier,
    inputs: List[Tensor],
    input_regions: List[Tuple[Tensor, Tensor]],
    target_g_t_constraints: List[List[List[Tuple[int, int, float]]]],
    sigmoid_encode_property: bool,
    batch_size: int,
    device: torch.device,
    timeout: float,
) -> Tuple[bool, Optional[List[np.ndarray]]]:  # is_verified, adv_example
    expanded_input_lbs: List[Tensor] = []
    expanded_input_ubs: List[Tensor] = []
    output_property_forms: List[OutputPropertyForm] = []

    print("filtering out easy instances with external batching...")

    for (input_lb, input_ub), gt_constraint in zip(
        input_regions, target_g_t_constraints
    ):

        if input_lb.dim() == len(parsed_config.input_dim):  # No batch dimension
            input_lb = input_lb.unsqueeze(0)
            input_ub = input_ub.unsqueeze(0)

        if sigmoid_encode_property:
            gt_constraint = get_sigmoid_gt_constraint(gt_constraint)

        # no unet (TODO: should this be supported?)

        output_property_form = OutputPropertyForm.create_from_properties(
            properties_to_verify=gt_constraint,
            disjunction_adapter=None,
            use_disj_adapter=parsed_config.verifier.outer.use_disj_adapter,  # TODO: assert that this is False?
            n_class=network.output_dim[-1],
            device=device,
            dtype=torch.get_default_dtype(),
        )

        assert (
            output_property_form.disjunction_adapter is None
        ), "proper disjunction not supported for batched pre-filtering"

        num_queries = output_property_form.property_matrix.shape[0]
        assert input_lb.shape == input_ub.shape
        assert input_lb.shape[0] in {1, num_queries}
        if input_lb.shape[0] == 1:
            expanded_input_lbs.append(input_lb.expand(num_queries, *input_lb.shape[1:]))
            expanded_input_ubs.append(input_ub.expand(num_queries, *input_ub.shape[1:]))
        else:
            expanded_input_lbs.append(input_lb)
            expanded_input_ubs.append(input_ub)

        output_property_forms.append(output_property_form)

    all_input_lbs = torch.cat(expanded_input_lbs)
    all_input_ubs = torch.cat(expanded_input_ubs)
    all_properties_matrix = torch.cat(
        tuple(
            output_property_form.property_matrix
            for output_property_form in output_property_forms
        )
    )
    all_properties_threshold = torch.cat(
        tuple(
            output_property_form.property_threshold
            for output_property_form in output_property_forms
        )
    )

    n_properties = all_properties_matrix.shape[0]

    assert all_input_lbs.shape[0] == n_properties
    assert all_input_ubs.shape[0] == n_properties
    assert all_properties_threshold.shape[0] == n_properties

    num_verified = 0
    num_falsified = 0

    def count_instances(tensor: Tensor) -> int:
        if tensor.dim() == 1:
            return int(tensor.sum().item())
        if tensor.dim() == 2:
            return int(tensor.all(1).sum().item())
        return -1  # TODO: replace with assertion; not worth crashing for printout atm

    def print_info(total: int) -> None:
        if num_verified != 0:
            print("verified: ", num_verified, " / ", total)
        if num_falsified != 0:
            print("falsified: ", num_falsified, " / ", total)
        if num_verified != total and num_falsified != total:
            print("remaining: ", total - num_verified - num_falsified, " / ", total)

    if batch_size > 0:
        if batch_size >= n_properties:

            subproblem_state = SubproblemState.create_default(
                split_state=None,
                optimize_prima=False,
                batch_size=n_properties,
                device=input_lb.device,
                use_params=False,
            )

            (
                verified,
                falsified,
                dp_lbs,
                _,
                _,
                _,
            ) = verifier._verify_query_with_deep_poly(
                all_input_lbs,
                all_input_ubs,
                all_properties_matrix,
                all_properties_threshold,
                compute_sensitivity=False,
                subproblem_state=subproblem_state,
                ibp_pass=verifier.optimizer.backsubstitution_config.box_pass,
            )

            num_verified = count_instances(verified)
            num_falsified = count_instances(falsified)
            print_info(n_properties)
        else:
            cur_index = 0
            all_verified: List[Tensor] = []
            all_falsified: List[Tensor] = []
            all_dp_lbs: List[Tensor] = []
            while cur_index < n_properties:
                next_index = min(cur_index + batch_size, n_properties)

                subproblem_state = SubproblemState.create_default(
                    split_state=None,
                    optimize_prima=False,
                    batch_size=next_index - cur_index,
                    device=input_lb.device,
                    use_params=False,
                )

                (
                    verified,
                    falsified,
                    dp_lbs,
                    _,
                    _,
                    _,
                ) = verifier._verify_query_with_deep_poly(
                    all_input_lbs[cur_index:next_index],
                    all_input_ubs[cur_index:next_index],
                    all_properties_matrix[cur_index:next_index],
                    all_properties_threshold[cur_index:next_index],
                    compute_sensitivity=False,
                    subproblem_state=subproblem_state,
                    ibp_pass=verifier.optimizer.backsubstitution_config.box_pass,
                )
                all_verified.append(verified)
                all_falsified.append(falsified)
                all_dp_lbs.append(dp_lbs)

                num_verified += count_instances(verified)
                num_falsified += count_instances(falsified)
                print_info(next_index)
                cur_index = next_index
            verified = torch.cat(all_verified)
            falsified = torch.cat(all_falsified)
            dp_lbs = torch.cat(all_dp_lbs)

        if verified.all():
            return True, None

        new_output_property_forms: List[OutputPropertyForm] = []

        cur_index = 0
        for output_property_form in output_property_forms:
            num_queries = output_property_form.property_matrix.shape[0]
            next_index = cur_index + num_queries
            new_output_property_form = output_property_form.update_properties_to_verify(
                verified[cur_index:next_index],
                falsified[cur_index:next_index],
                dp_lbs[cur_index:next_index],
                true_ub=False,
            )
            new_output_property_forms.append(new_output_property_form)
            cur_index = next_index
    else:
        new_output_property_forms = output_property_forms

    if verifier.outer.input_domain_splitting:
        queue: List[
            Tuple[
                Tensor,
                Tensor,
                Tuple[Tensor, Tensor, Tensor],
                int,
                Optional[Sequence[Sequence[Tuple[int, int, float]]]],
            ]
        ] = []
        for input_point, input_lb, input_ub, output_property_form in zip(
            inputs, expanded_input_lbs, expanded_input_ubs, new_output_property_forms
        ):
            if not output_property_form.properties_to_verify:
                continue
            queue.append(
                (
                    input_lb,
                    input_ub,
                    (
                        output_property_form.property_matrix,
                        output_property_form.property_threshold,
                        output_property_form.combination_matrix,
                    ),
                    verifier.domain_splitting.max_depth,
                    output_property_form.properties_to_verify,
                )
            )
        queue, _ = verifier._conduct_input_domain_splitting(
            verifier.domain_splitting, queue, timeout, None
        )
        if len(queue) == 0:
            return True, None
        elif len(queue) == 1 and queue[0][-1] == -1:
            # counterexample region returned
            adv_example = queue[0][0]
            out = verifier.network(adv_example)
            assert (input_lb <= adv_example).__and__(input_ub >= adv_example).all()
            if not _evaluate_cstr(queue[0][-1], out.detach(), torch_input=True):
                print("Adex found via splitting")
                return False, [np.array(adv_example.cpu())]
            else:
                assert False, "should have been a counterexample"
        else:
            pass

    for input_point, input_lb, input_ub, output_property_form in zip(
        inputs, expanded_input_lbs, expanded_input_ubs, new_output_property_forms
    ):
        (
            is_verified,
            adv_example,
            lower_idx,
            lower_bound_tmp,
            upper_bound_tmp,
        ) = verifier.verify_via_config(
            0,
            input_point,
            input_lb,
            input_ub,
            output_property_form,
            timeout=timeout,
        )

        assert output_property_form.disjunction_adapter is None

        if adv_example is not None:
            assert not is_verified
            break
        if not is_verified:
            break
    return is_verified, adv_example


def check_sigmoid_tanh_tangent(network: AbstractNetwork) -> None:
    has_sig_layer = False
    has_tanh_layer = False
    for tag, layer in network.layer_id_to_layer.items():
        if isinstance(layer, Sigmoid):
            has_sig_layer = True
        if isinstance(layer, Tanh):
            has_tanh_layer = True

    if has_sig_layer:
        (
            Sigmoid.intersection_points,
            Sigmoid.tangent_points,
            Sigmoid.step_size,
            Sigmoid.max_x,
        ) = SigBase._compute_bound_to_tangent_point(sig, d_sig)

    if has_tanh_layer:
        (
            Tanh.intersection_points,
            Tanh.tangent_points,
            Tanh.step_size,
            Tanh.max_x,
        ) = SigBase._compute_bound_to_tangent_point(tanh, d_tanh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare verification instances on the vnn22 datasets. Simply provide the corresponding nets and specs"
    )
    parser.add_argument(
        "-b", "--benchmark", type=str, help="The benchmark id", required=True
    )
    parser.add_argument(
        "-n",
        "--netname",
        type=str,
        help="The network path",
        required=True,
    )
    parser.add_argument(
        "-s",
        "--vnnlib_spec",
        type=str,
        help="The vnnlib spec path",
        required=True,
    )
    parser.add_argument(
        "-r",
        "--results_path",
        type=str,
        help="",
        required=True,
    )
    parser.add_argument(
        "-t",
        "--timeout",
        type=float,
        help="",
        required=True,
    )

    args = parser.parse_args()

    run_instance(
        args.benchmark,
        args.netname,
        args.vnnlib_spec,
        args.results_path,
        int(args.timeout),
    )
