import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import dill  # type: ignore[import]
import numpy as np
import onnx  # type: ignore[import]
import torch
import torch.nn as nn
from bunch import Bunch  # type: ignore[import]
from torch import Tensor

from src.abstract_layers.abstract_network import AbstractNetwork
from src.abstract_layers.abstract_sequential import Sequential
from src.utilities.argument_parsing import get_config_from_json
from src.utilities.config import Config, Dtype, make_config
from src.utilities.loading.dnnv_simplify import simplify_onnx
from src.utilities.loading.network import (
    freeze_network,
    load_onnx_from_proto,
    load_onnx_model,
)
from src.utilities.loading.vnn_spec_loader import (  # translate_constraints_to_label,
    parse_vnn_lib_prop,
)

FILE_DIR = os.path.realpath(os.path.dirname(__file__))

NET_TO_CONFIG_MAP = os.path.realpath(
    os.path.join(FILE_DIR, "../..", "configs/net_to_config.csv")
)
META_CONFIG = os.path.realpath(
    os.path.join(FILE_DIR, "../..", "configs/meta_config.json")
)
TEMP_RUN_DIR = os.path.realpath(os.path.join(FILE_DIR, "../..", "run"))


def generate_constraints(
    class_num: int, y: int
) -> List[List[List[Tuple[int, int, float]]]]:
    return [[[(y, i, 0)] for i in range(class_num) if i != y]]


def get_io_constraints_from_spec(
    spec_path: str, config: Config, device: torch.device
) -> Tuple[
    List[Tensor], List[Tuple[Tensor, Tensor]], List[List[List[Tuple[int, int, float]]]]
]:
    dtype = np.float32 if config.dtype == Dtype("float32") else np.float64

    if not os.path.exists(spec_path) and not spec_path.endswith(".gz"):
        spec_path += ".gz"

    input_boxes, output_gt_constraints = parse_vnn_lib_prop(spec_path, dtype=dtype)
    input_regions = []
    inputs = []
    for box in input_boxes:
        input_lb_arr, input_ub_arr = box
        input_lb, input_ub = (
            torch.from_numpy(input_lb_arr).to(device),
            torch.from_numpy(input_ub_arr).to(device),
        )
        input_lb, input_ub = input_lb.reshape(config.input_dim), input_ub.reshape(
            config.input_dim
        )
        input_regions.append((input_lb, input_ub))
        inputs.append((input_lb + input_ub) / 2)

    return inputs, input_regions, output_gt_constraints


class NetworkConfig:
    default_config: Optional[Dict[str, str]] = None
    specific_config: Dict[str, Dict[str, str]]

    def __init__(self, config: Bunch):
        self.specific_config: Dict[str, Dict[str, str]] = {}
        for id, cfg in config.items():
            self.specific_config[id] = cfg
            if id == "default":
                self.default_config = cfg

    def get_config_given_characteristics(self, net_char: str) -> Dict[str, str]:

        if net_char in self.specific_config:
            print(f"Using characteristic: {net_char}")
            return self.specific_config[net_char]
        else:
            assert self.default_config is not None
            print("WARNING! Using default config.")
            return self.default_config


class MetaConfig:
    config_mapping: Dict[str, NetworkConfig]

    def __init__(self, meta_config: Bunch):
        self.config_mapping = {}
        for k, v in meta_config.items():
            if not "default" in v.keys():
                v.update(meta_config["default"])
            self.config_mapping[k] = NetworkConfig(Bunch(**v))

    def get_benchmark_mapping(self, benchmark: str) -> NetworkConfig:
        if benchmark in self.config_mapping:
            return self.config_mapping[benchmark]
        else:
            return self.config_mapping["default"]


def load_meta_config() -> MetaConfig:
    assert os.path.exists(META_CONFIG), "META_CONFIG path does not exist."
    with open(META_CONFIG) as f:
        meta_conf = json.load(f)
        meta_config = MetaConfig(Bunch(**meta_conf))
    return meta_config


def get_input_characteristics(net: nn.Sequential, onnx_shape: Tuple[int, ...]) -> None:
    pass


def get_network_characteristics(net: Sequential) -> str:
    act_layers = net.get_activation_layers()

    node_count = 0
    for layer in act_layers.values():
        node_count += np.prod(layer.output_dim)

    param_count = sum(p.numel() for p in net.parameters())

    return f"{node_count}-{param_count}"


def get_asnet_from_path(net_path: str, device: torch.device) -> None:
    net_format = net_path.split(".")[-1]
    if net_format in ["onnx", "gz"]:
        net, onnx_shape, inp_name = load_onnx_model(net_path)  # Like this for mypy
    else:
        assert False, f"No net loaded for net format: {net_format}."

    as_net = AbstractNetwork.from_concrete_module(net, onnx_shape)
    node_count = get_network_characteristics(as_net)
    print(f"{net_path} char: {node_count}")


def get_net_asnet_conf(
    benchmark_id: str, net_path: str, device: torch.device
) -> Tuple[nn.Module, AbstractNetwork, str, Bunch, Config]:
    # network
    net_format = net_path.split(".")[-1]
    if net_format in ["onnx", "gz"]:
        net, onnx_shape, inp_name = load_onnx_model(net_path)  # Like this for mypy
    else:
        assert False, f"No net loaded for net format: {net_format}."

    as_net = AbstractNetwork.from_concrete_module(net, onnx_shape)

    meta_config = load_meta_config()
    node_count = get_network_characteristics(as_net)
    benchmark_meta_config = meta_config.get_benchmark_mapping(benchmark_id)
    config_obj = benchmark_meta_config.get_config_given_characteristics(node_count)
    config_path = os.path.realpath(
        os.path.join(FILE_DIR, "../..", config_obj["config"])
    )
    print(f"== Using config {config_path} ==")

    json_config = get_config_from_json(config_path)
    parsed_config = make_config(**json_config)

    if ("adapt_input_dim" in config_obj) or (benchmark_id not in net_path):
        print(f"Setting shape: {onnx_shape}")
        json_config["input_dim"] = onnx_shape
        parsed_config.input_dim = onnx_shape

    net.to(device)
    net.eval()
    freeze_network(net)

    if parsed_config.dtype == Dtype.float64:
        torch.set_default_dtype(torch.float64)
        net = net.double()
    else:
        torch.set_default_dtype(torch.float32)
        net = net.float()

    if parsed_config.verifier.outer.simplify_onnx:
        assert onnx_shape is not None
        assert inp_name is not None
        # export current model to onnx for dtype
        try:
            temp_dir = os.path.realpath(os.path.join(FILE_DIR, "../.." "temp_convert"))
            net_pref = "simplify"
            onnx_path = f"{temp_dir}/{net_pref}.onnx"
            Path(temp_dir).mkdir(parents=True, exist_ok=True)
            x = torch.rand((1, *onnx_shape), device=device)
            torch.onnx.export(
                net,
                x,
                onnx_path,
                export_params=True,
                training=torch.onnx.TrainingMode.EVAL,
                do_constant_folding=True,
                verbose=False,
                input_names=[inp_name],
                output_names=["output"],
            )
            onnx_model = onnx.load(onnx_path)
            onnx_model = simplify_onnx(onnx_model)
            net_new, _, _ = load_onnx_from_proto(onnx_model, net_path)
            net_new.to(device)
            net_new.eval()
            freeze_network(net_new)
            net = net_new
        except Exception as e:
            print("Exception simplifying onnx model: ", e)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    assert isinstance(net, nn.Sequential)

    return net, as_net, config_path, json_config, parsed_config


def create_instance_from_vnn_spec(
    benchmark_name: str, net_path: str, spec_path: str
) -> None:

    shutil.rmtree(f"{TEMP_RUN_DIR}", ignore_errors=True)

    # net_path = os.path.realpath(os.path.join(FILE_DIR, "../../..", "vnncomp2022_benchmarks", net_path))
    # spec_path = os.path.realpath(os.path.join(FILE_DIR, "../../..", "vnncomp2022_benchmarks",spec_path))

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    net, as_network, config_path, json_config, parsed_config = get_net_asnet_conf(
        benchmark_name, net_path, device
    )
    # Get data
    (inputs, input_regions, target_g_t_constraints) = get_io_constraints_from_spec(
        spec_path=spec_path, config=parsed_config, device=device
    )

    # Write out
    Path(f"{TEMP_RUN_DIR}").mkdir(parents=False, exist_ok=False)

    # 1. config
    with open(f"{TEMP_RUN_DIR}/config.json", "w") as f:
        json.dump(json_config, f, indent=2)

    # Write out abstract network
    torch.save(as_network, f"{TEMP_RUN_DIR}/abs_network.onnx")

    # 3. Write out input
    input_path = f"{TEMP_RUN_DIR}/inputs"
    Path(input_path).mkdir(parents=True, exist_ok=False)
    conc_inputs = torch.stack(inputs, 0)
    torch.save(conc_inputs, f"{input_path}/inputs.pt")

    input_region_path = f"{TEMP_RUN_DIR}/input_regions"
    Path(input_region_path).mkdir(parents=True, exist_ok=False)
    stack_lb = torch.stack([lb for (lb, ub) in input_regions], 0)
    stack_ub = torch.stack([ub for (lb, ub) in input_regions], 0)
    torch.save(stack_lb, f"{input_region_path}/input_lbs.pt")
    torch.save(stack_ub, f"{input_region_path}/input_ubs.pt")

    target_g_t_constraint_path = f"{TEMP_RUN_DIR}/io_constraints"
    Path(target_g_t_constraint_path).mkdir(parents=True, exist_ok=False)
    with open(f"{target_g_t_constraint_path}/target_g_t_constraints.pkl", "wb") as file:
        dill.dump(target_g_t_constraints, file)

    # 4. Write METADATA
    metadata_path = f"{TEMP_RUN_DIR}/metadata.txt"
    with open(f"{metadata_path}", "w") as file:
        file.write(f"benchmark: {benchmark_name} \n")
        file.write(f"network_path: {net_path}\n")
        file.write(f"spec_path: {spec_path}\n")
        file.write(f"config_path: {config_path}\n")
        file.write(f"Simplified ONNX: {parsed_config.verifier.outer.simplify_onnx}\n")
        file.write(f"inputs: {len(inputs)}\n")
        file.write(f"input_regions: {len(input_regions)}\n")
        file.write(f"target_gts: {len(target_g_t_constraints)}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare verification instances on the vnn22 datasets. Simply provide the corresponding net and spec"
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
    args = parser.parse_args()

    create_instance_from_vnn_spec(args.benchmark, args.netname, args.vnnlib_spec)
