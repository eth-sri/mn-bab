import argparse
import csv
import os
import re
import shutil
import time
from cmath import inf
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from comet_ml import Experiment  # type: ignore[import]

import numpy as np
import onnx  # type: ignore[import]
import torch
import torch.nn as nn
from bunch import Bunch  # type: ignore[import]
from torch import Tensor

from src.abstract_layers.abstract_network import AbstractNetwork
from src.abstract_layers.abstract_sigmoid import Sigmoid
from src.mn_bab_verifier import MNBaBVerifier
from src.utilities.argument_parsing import get_config_from_json
from src.utilities.config import Config, Dtype, make_config
from src.utilities.initialization import seed_everything
from src.utilities.loading.dnnv_simplify import simplify_onnx
from src.utilities.loading.network import (
    freeze_network,
    load_net,
    load_onnx_from_proto,
    load_onnx_model,
)
from src.utilities.loading.vnn_spec_loader import (  # translate_constraints_to_label,
    parse_vnn_lib_prop,
)
from src.utilities.prepare_instance import get_network_characteristics
from src.utilities.output_property_form import OutputPropertyForm

NET_TO_CONFIG_MAP = "configs/net_to_config.csv"


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


def get_asnet(
    net_path: Optional[str], config: Config, device: torch.device
) -> Tuple[nn.Module, AbstractNetwork]:
    # Get bounds
    if net_path is None:
        net_path = config.network.path
    if net_path.endswith(".gz"):
        net_path = ".".join(net_path.split(".")[:-1])
    net_format = net_path.split(".")[-1]
    if net_format in ["onnx", "gz"]:
        net_seq, onnx_shape, inp_name = load_onnx_model(net_path)  # Like this for mypy
        net: nn.Module = net_seq
    elif net_format == "pt" or net_format == "pth" or net_format == "pyt":
        net = load_net(**config.network.load_params())
    else:
        assert False, f"No net loaded for net format: {net_format}."

    net.to(device)
    net.eval()
    freeze_network(net)

    if config.dtype == Dtype.float64:
        torch.set_default_dtype(torch.float64)
        net = net.double()
    else:
        torch.set_default_dtype(torch.float32)
        net = net.float()

    if len(config.input_dim) == 0:
        print(f"Setting shape: {onnx_shape}")
        config.input_dim = onnx_shape

    if config.verifier.outer.simplify_onnx:
        assert onnx_shape is not None
        assert inp_name is not None
        # export current model to onnx for dtype
        try:
            temp_dir = "temp_convert"
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
            # for i in range(50):
            #     x = torch.rand((1, *onnx_shape), device=device)
            #     out_old = net(x)
            #     out = net_new(x)
            #     assert torch.isclose(out_old, out, atol=1e-7).all()
            net = net_new
        except Exception as e:
            print("Exception simplifying onnx model: ", e)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    assert isinstance(net, nn.Sequential)
    as_net = AbstractNetwork.from_concrete_module(net, config.input_dim).to(device)
    freeze_network(as_net)
    return net, as_net


def generate_adv_output_str(
    adv_example: np.ndarray,
    network: AbstractNetwork,
    input_shape: Tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> str:
    adv_tensor = torch.tensor(adv_example, device=device, dtype=dtype)
    adv_out: Tensor = network(adv_tensor)

    adv_tensor = adv_tensor.flatten()
    adv_out = adv_out.flatten()
    out_enc = "("
    for i in range(adv_tensor.numel()):
        out_enc += f"(X_{i} {adv_tensor[i]})\n"
    for j in range(adv_out.numel()):
        out_enc += f"(Y_{j} {adv_out[j]})\n"
    out_enc = out_enc[:-1]
    out_enc += ")\n"

    return out_enc


class VerificationInstance:
    def __init__(
        self,
        network: Union[AbstractNetwork, str],
        verifier: Optional[MNBaBVerifier],
        config: Config,
        inputs: Union[List[Tensor], str],
        input_regions: Optional[List[Tuple[Tensor, Tensor]]],
        target_gt_constraints: Optional[List[List[List[Tuple[int, int, float]]]]],
    ) -> None:

        self.network: Optional[Union[AbstractNetwork, str]] = network
        self.verifier: Optional[MNBaBVerifier] = verifier
        self.inputs = inputs
        self.input_constraints = input_regions
        self.config = config

        self.target_constr = target_gt_constraints
        self.is_verified: bool = False
        self.has_adv_example: bool = False
        self.lower_idx: int = (
            -1
        )  # Index at which our computation timed out (for comparison)
        self.time: float = inf
        self.lower_bound: float = inf

    @classmethod
    def create_instance_from_vnn_spec(
        cls, net_path: str, spec_path: str, config: Bunch
    ) -> "VerificationInstance":

        # Get Config
        parsed_config = make_config(**config)
        seed_everything(parsed_config.random_seed)

        experiment_logger = Experiment(**parsed_config.logger.comet_options)
        experiment_logger.set_name(parsed_config.experiment_name)
        experiment_logger.log_parameters(parsed_config)

        if torch.cuda.is_available() and parsed_config.use_gpu:
            device = torch.device("cuda")
            experiment_logger.log_text("Using gpu")
            experiment_logger.log_text(torch.cuda.get_device_name(0))
        else:
            device = torch.device("cpu")
            experiment_logger.log_text("Using cpu")

        # Get data

        if parsed_config.load_eager:  # TODO @Robin
            (
                inputs,
                input_regions,
                target_g_t_constraints,
            ) = get_io_constraints_from_spec(
                spec_path=spec_path, config=parsed_config, device=device
            )
            net, as_network = get_asnet(net_path, parsed_config, device)
            # Get verifier
            verifier = MNBaBVerifier(as_network, device, parsed_config.verifier)

            return cls(
                as_network,
                verifier,
                parsed_config,
                inputs,
                input_regions,
                target_g_t_constraints,
            )
        else:  # Load the network and specs when running the instance
            return cls(
                net_path,
                None,
                parsed_config,
                spec_path,
                None,
                None,
            )

    @classmethod
    def create_instance_from_input_and_eps(
        cls,
        net_path: Optional[str],
        input: Tensor,
        eps: float,
        target_label: int,
        config_path: str,
        num_classes: int,
    ) -> "VerificationInstance":

        # Get Config
        config_file = get_config_from_json(config_path)
        config = make_config(**config_file)

        if torch.cuda.is_available() and config.use_gpu:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        # Get data
        input = input.to(device)
        input_lb = torch.clamp(input - eps, 0, 1)
        input_ub = torch.clamp(input + eps, 0, 1)

        net, as_net = get_asnet(net_path, config, device)

        # Get verifier
        verifier = MNBaBVerifier(as_net, device, config.verifier)

        return cls.create_instance_for_batch_ver(as_net, verifier, input, input_lb, input_ub, target_label, config, num_classes)


    @classmethod
    def create_instance_for_batch_ver(
        cls,
        as_net: AbstractNetwork,
        verifier: MNBaBVerifier,
        input: Tensor,
        input_lb: Tensor,
        input_ub: Tensor,
        target_label: int,
        config: Config,
        num_classes: int,
    ) -> "VerificationInstance":

        # Get Config
        seed_everything(config.random_seed)

        experiment_logger = Experiment(**config.logger.comet_options)
        experiment_logger.set_name(config.experiment_name)
        experiment_logger.log_parameters(config)

        if torch.cuda.is_available() and config.use_gpu:
            experiment_logger.log_text("Using gpu")
            experiment_logger.log_text(torch.cuda.get_device_name(0))
        else:
            if config.use_gpu:
                print("WARNING: GPU NOT AVAILABLE")
            experiment_logger.log_text("Using cpu")

        # Get data
        target_gt_constraints = generate_constraints(num_classes, target_label)

        return cls(
            as_net,
            verifier,
            config,
            [input],
            [(input_lb, input_ub)],
            target_gt_constraints,
        )


    def run_instance(self) -> None:  # noqa: C901

        device = torch.device("cuda") if self.config.use_gpu else torch.device("cpu")

        if isinstance(self.network, str):
            net, as_net = get_asnet(self.network, self.config, device)
            self.network = as_net
        assert isinstance(self.network, AbstractNetwork)

        if isinstance(self.inputs, str):
            (
                self.inputs,
                self.input_constraints,
                self.target_constr,
            ) = get_io_constraints_from_spec(
                spec_path=self.inputs, config=self.config, device=device
            )

        assert isinstance(self.inputs, List) and len(self.inputs) > 0
        assert self.input_constraints is not None and len(self.input_constraints) > 0
        assert self.target_constr is not None

        n_regions = len(self.input_constraints)
        n_non_zero_width = max(
            [(lb != ub).sum().item() for lb, ub in self.input_constraints]
        )

        node_count = get_network_characteristics(self.network)
        print(
            f"Analyzing {n_regions} input regions with at most {n_non_zero_width}/{self.inputs[0].numel()} non-zero-width inputs and neurons-parameters: {node_count}"
        )

        sigmoid_encode_property: bool = False
        if isinstance(self.network.layers[-1], Sigmoid):
            self.network.layers = self.network.layers[:-1]
            self.network.set_activation_layers()
            sigmoid_encode_property = True

        if self.config.dtype == Dtype.float64:
            torch.set_default_dtype(torch.float64)
            self.network = self.network.double()
            self.inputs = [x.double() for x in self.inputs]
            self.input_constraints = [
                (x[0].double(), x[1].double()) for x in self.input_constraints
            ]
        else:
            torch.set_default_dtype(torch.float32)
            self.inputs = [x.float() for x in self.inputs]
            self.input_constraints = [
                (x[0].float(), x[1].float()) for x in self.input_constraints
            ]

        if self.verifier is None:
            assert isinstance(self.network, AbstractNetwork)
            self.verifier = MNBaBVerifier(self.network, device, self.config.verifier)

        # assert self.verifier is not None
        start_time = time.time()
        lower_bound = float("inf")  # This is intentionally set to positive infinity
        upper_bound = float("inf")

        # TODO here we could order input region/constraint pairings to find counterexamples quicker

        for input_point, (input_lb, input_ub), gt_constraint in zip(
            self.inputs, self.input_constraints, self.target_constr
        ):

            if time.time() - start_time > self.config.timeout:
                is_verified = False
                adv_example = None
                break

            if input_lb.dim() == len(self.config.input_dim):  # No batch dimension
                input_lb = input_lb.unsqueeze(0)
                input_ub = input_ub.unsqueeze(0)
                input_point = input_point.unsqueeze(0)
            assert tuple(input_lb.shape[1:]) == self.config.input_dim

            if sigmoid_encode_property:
                gt_constraint = get_sigmoid_gt_constraint(gt_constraint)

            if "unet" in self.config.benchmark_instances_path:
                gt_constraint, gt_target = get_unet_gt_constraint(
                    input_point, (input_lb, input_ub), gt_constraint
                )

            assert isinstance(self.network, AbstractNetwork)
            out_prop_form = OutputPropertyForm.create_from_properties(
                gt_constraint,
                None,
                self.config.verifier.outer.use_disj_adapter,
                self.network.output_dim[-1],
                device,
                torch.get_default_dtype(),
            )

            if "unet" in self.config.benchmark_instances_path:
                assert gt_target
                (
                    is_verified,
                    adv_example,
                    lower_idx,
                    lower_bound_tmp,
                    upper_bound_tmp,
                ) = self.verifier.verify_unet_via_config(
                    0,
                    input_point,
                    input_lb,
                    input_ub,
                    out_prop_form,
                    verification_target=gt_target,
                    timeout=self.config.timeout + start_time,
                )

            else:

                if out_prop_form.disjunction_adapter is not None:
                    self.verifier.append_out_adapter(
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
                ) = self.verifier.verify_via_config(
                    0,
                    input_point,
                    input_lb,
                    input_ub,
                    out_prop_form,
                    timeout=self.config.timeout + start_time,
                )

                if out_prop_form.disjunction_adapter is not None:
                    self.verifier.remove_out_adapter()

                if lower_bound_tmp is not None and upper_bound_tmp is not None:
                    lower_bound = min(
                        lower_bound_tmp, lower_bound
                    )  # taking the smallest/worst lower bound over input regions
                    upper_bound = min(
                        upper_bound_tmp, upper_bound
                    )  # taking the smallest/worst counter example over regions
                if adv_example is not None:
                    assert not is_verified
                    _ = generate_adv_output_str(
                        adv_example[0],
                        self.verifier.network,
                        input_lb.shape,
                        device,
                        torch.get_default_dtype(),
                    )
                    break
                if not is_verified:
                    break

        total_time = time.time() - start_time

        self.is_verified = is_verified
        self.adv_example = adv_example

        print(f"Total instance time: {total_time}")

        self.time = total_time
        if is_verified:
            self.lower_bound = 0
        elif adv_example is None and lower_bound:
            self.lower_bound = lower_bound

    def free_memory(self) -> None:
        self.network = None
        self.verifier = None
        self.input_constraints = None  # type: ignore[assignment]
        self.target_constr = None  # type: ignore[assignment]
        self.input = None  # type: ignore[assignment]
        self.input_lb = None  # type: ignore[assignment]
        self.input_ub = None  # type: ignore[assignment]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_instance_dict(
    inst_dict: Dict[Tuple[str, str], List[VerificationInstance]]
) -> None:
    verified = 0
    disproved = 0
    verif_inst: List[Tuple[int, float]] = []
    disproved_inst: List[Tuple[int, float]] = []
    for i, (inst_id, inst) in enumerate(inst_dict.items()):
        print(
            f"\n Running instanceset {i} - {inst_id} - Timeout: {inst[0].config.timeout}"
        )
        is_verif = run_instances(inst)

        if is_verif > 0:
            for cfg in inst:
                if cfg.is_verified:
                    verif_inst.append((i, cfg.time))

        if is_verif < len(inst):
            for cfg in inst:
                if cfg.adv_example is not None:
                    disproved_inst.append((i, cfg.time))
                    disproved += 1

        verified += is_verif

    for i, t in verif_inst:
        print(f"Verified instance: {i} Time: {t}")
    for i, t in disproved_inst:
        print(f"Disproved instance: {i} Time: {t}")
    print(
        f"Verified: {verified} Disproved: {disproved} out of {len(inst_dict.values())}"
    )


def run_instances(instances: List[VerificationInstance]) -> int:
    verified = 0
    for i, inst in enumerate(instances):
        print("=" * 20)
        print(f"Running config {i}")
        inst.run_instance()
        if inst.is_verified:
            verified += 1
            print(f"Verified - Time: {inst.time}")
        else:
            print("Not verified")
        inst.free_memory()
    return verified


def get_val_from_re_dict(key: str, dict: Dict[str, str]) -> str:
    for k, v in dict.items():
        if re.match(k, key):
            return v
    print("WARNING! USING DEFAULT CONFIG")
    return "configs/vnncomp22/default_config.json"
    raise KeyError


def create_instances_from_args(
    args: Bunch,
) -> Dict[Tuple[str, str], List[VerificationInstance]]:

    instance_dict: Dict[Tuple[str, str], List[VerificationInstance]] = {}
    n2c_map: Dict[str, str] = {}
    use_n2c_map = False

    # All information in the config file
    if args.instances is None and args.nets is None:
        assert len(args.configs) > 0
        for config in args.configs:
            bunch_config = get_config_from_json(args.configs[0])
            # Load instance file from config
            inst = bunch_config.benchmark_instances_path
            pref = "/".join(inst.split("/")[:-1])
            with open(inst, "r") as f:
                lines = f.readlines()
                for line in lines:
                    net, spec, timeout = line.strip().split(",")
                    onnx_path = os.path.join(pref, net.strip())
                    spec_path = os.path.join(pref, spec.strip())
                    bunch_config.timeout = int(float(timeout.strip()))
                    bunch_config.network_path = onnx_path

                    v_inst = VerificationInstance.create_instance_from_vnn_spec(
                        onnx_path, spec_path, bunch_config
                    )
                    if (net, spec) in instance_dict:
                        instance_dict[(net, spec)].append(v_inst)
                    else:
                        instance_dict[(net, spec)] = [v_inst]

    elif (
        args.instances
    ):  # Run set of instances with certain configs - does not need a benchmark path in the config
        if args.configs is not None:
            assert len(args.instances) == len(args.configs)
            print("Using one config per instance file")
        else:
            print(f"Loading configs for known nets from {NET_TO_CONFIG_MAP}")
            with open(NET_TO_CONFIG_MAP, "r") as f:
                n2c = csv.reader(f)
                for k, v in n2c:
                    n2c_map[k.strip()] = v.strip()
            use_n2c_map = True
        for i, inst in enumerate(args.instances):
            print(f"Loading the following instances {inst}")
            pref = "/".join(inst.split("/")[:-1])
            with open(inst, "r") as f:
                lines = f.readlines()
                for line in lines:
                    net, spec, timeout = line.strip().split(",")
                    onnx_path = os.path.join(pref, net.strip())
                    spec_path = os.path.join(pref, spec.strip())

                    if use_n2c_map:
                        inst_pref = "/".join(inst.split("/")[:-1]) + "/"
                        cfg_path = get_val_from_re_dict(inst_pref + net, n2c_map)
                        config = get_config_from_json(cfg_path)
                    else:
                        config = get_config_from_json(args.configs[i])
                    config.timeout = int(float(timeout.strip()))
                    config.network_path = onnx_path

                    v_inst = VerificationInstance.create_instance_from_vnn_spec(
                        onnx_path, spec_path, config
                    )
                    if (net, spec) in instance_dict:
                        instance_dict[(net, spec)].append(v_inst)
                    else:
                        instance_dict[(net, spec)] = [v_inst]

    else:
        assert len(args.nets) == 1
        if len(args.configs) == 1:
            config = get_config_from_json(args.configs[0])
            for spec in args.specs:
                onnx_path = args.nets[0]
                spec_path = spec
                config.timeout = timeout
                v_inst = VerificationInstance.create_instance_from_vnn_spec(
                    onnx_path, spec_path, config
                )
                if (net, spec) in instance_dict:
                    instance_dict[(net, spec)].append(v_inst)
                else:
                    instance_dict[(net, spec)] = [v_inst]
        else:
            assert len(args.configs) == len(args.specs)
            for spec, config in zip(args.specs, args.configs):
                onnx_path = args.nets[0]
                spec_path = spec
                config.timeout = timeout
                v_inst = VerificationInstance.create_instance_from_vnn_spec(
                    onnx_path, spec_path, config
                )
                if (net, spec) in instance_dict:
                    instance_dict[(net, spec)].append(v_inst)
                else:
                    instance_dict[(net, spec)] = [v_inst]
    print(config)
    return instance_dict


def rank_instances(instances: List[VerificationInstance]) -> List[VerificationInstance]:

    verified_instances = [i for i in instances if i.is_verified]
    non_verified_instances = [i for i in instances if not i.is_verified]
    adv_instances = [i for i in instances if i.has_adv_example]
    assert not (
        len(verified_instances) > 0 and len(adv_instances)
    ), "Verified and adverserial"

    verified_instances.sort(key=lambda x: x.time)
    non_verified_instances.sort(key=lambda x: x.lower_bound, reverse=True)
    if len(adv_instances) > 0:
        return adv_instances + non_verified_instances + verified_instances
    elif len(verified_instances) > 0:
        return verified_instances + non_verified_instances
    else:
        return verified_instances


def get_unet_gt_constraint(
    input_point: Tensor,
    input_bounds: Tuple[Tensor, Tensor],
    gt_constraint: List[List[Tuple[int, int, float]]],
) -> Tuple[List[List[Tuple[int, int, float]]], int]:
    mask = input_point[0, 3, :]

    assert (mask == input_bounds[0][0, 3, :]).all()
    assert (mask == input_bounds[1][0, 3, :]).all()

    mask = mask.flatten()
    flat_dim = mask.numel()
    new_consts: List[List[Tuple[int, int, float]]] = []
    for i in range(flat_dim):
        if mask[i] == 1.0:  # dim 1 should be larger
            new_consts.append([(i + flat_dim, i, 0)])
        elif mask[i] == 0.0:
            new_consts.append([(i, i + flat_dim, 0)])
        else:
            assert "Unknown mask entry"

    assert len(gt_constraint) == 1
    assert gt_constraint[0][0][0] == 0 and gt_constraint[0][0][1] == -1

    return new_consts, int(gt_constraint[0][0][2])


def get_sigmoid_gt_constraint(
    gt_constraint: List[List[Tuple[int, int, float]]],
) -> List[List[Tuple[int, int, float]]]:

    new_consts: List[List[Tuple[int, int, float]]] = []
    for disj_clause in gt_constraint:
        new_clause: List[Tuple[int, int, float]] = []
        for atom in disj_clause:
            if atom[0] == -1:
                new_atom = (-1, atom[1], np.log(atom[2] / (1 - atom[2])))
            elif atom[1] == -1:
                new_atom = (atom[0], -1, np.log(atom[2] / (1 - atom[2])))
            else:
                assert atom[2] == 0
                new_atom = atom
            assert not (new_atom[0] == -1 and new_atom[1] == -1)
            new_clause.append(new_atom)
        new_consts.append(new_clause)

    return new_consts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run verification instances on the vnn21 (& vnn22) datasets. Simply provide the corresponding nets, specs, and configs"
    )
    parser.add_argument(
        "-is",
        "--instances",
        type=str,
        nargs="*",
        help="Loads one or multiple instance files. Overrides any manually given nets and specs given.",
    )
    parser.add_argument(
        "-n",
        "--nets",
        type=str,
        nargs="*",
        help="relative path for each net. In case it is a directory path we load all corresponding nets in the directory.",
    )
    parser.add_argument(
        "-s",
        "--specs",
        type=str,
        nargs="*",
        help="The specs corresponding to the nets. In case it is a directory path we load all corresponding specs in the directory.",
    )
    parser.add_argument(
        "-c",
        "--configs",
        type=str,
        nargs="*",
        help="The configs corresponding to the nets x specs. Either we load a single config for all specs or one config for each spec",
    )
    parser.add_argument(
        "-in",
        "--inputs",
        type=str,
        nargs="*",
        help="The input file we want to load examples from. Used for finding epsilon values on nets",
    )
    parser.add_argument(
        "-t",
        "--timeout",
        type=int,
        default=400,
        help="The timeout to apply per instance in case we do not load instances from a csv file.",
    )
    parser.add_argument(
        "-sub",
        "--subsample",
        type=int,
        default=1,
        help="Subsample each nets specs by the given factor for faster evaluation.",
    )
    parser.add_argument(
        "-ni",
        "--num_images",
        type=int,
        default=100,
        help="How many images to load from the image source. Defaults to 100.",
    )

    args = parser.parse_args()

    instances = create_instances_from_args(args)
    print(f"Running {len(instances)} instances.")
    run_instance_dict(instances)
