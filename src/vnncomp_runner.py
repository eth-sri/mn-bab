import os
import time

from comet_ml import Experiment  # type: ignore[import]

import torch
from torch import nn

from src.abstract_layers.abstract_network import AbstractNetwork
from src.mn_bab_verifier import MNBaBVerifier
from src.utilities.argument_parsing import get_args, get_config_from_json
from src.utilities.config import make_verifier_config
from src.utilities.initialization import seed_everything
from src.utilities.loading.network import freeze_network, load_net
from src.utilities.vnncomp_input_parsing import (
    parse_vnn_lib_prop,
    translate_box_to_sample,
    translate_constraints_to_label,
)

if __name__ == "__main__":
    args = get_args()
    config = get_config_from_json(args.config)
    seed_everything(config.random_seed)

    experiment_logger = Experiment(
        api_key=config.comet_api_key,
        project_name=config.comet_project_name,
        workspace=config.comet_workspace,
        disabled=not config.use_online_logging,
    )
    experiment_logger.set_name(config.experiment_name)
    experiment_logger.log_parameters(config)

    if torch.cuda.is_available() and config.use_gpu:
        device = torch.device("cuda")
        experiment_logger.log_text("Using gpu")
        experiment_logger.log_text(torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        experiment_logger.log_text("Using cpu")

    benchmark_instances_file = config.benchmark_instances_path

    with open(benchmark_instances_file, "r") as f:
        lines = f.readlines()
    instances = [[x.strip() for x in line.split(",")] for line in lines]
    instance_dir = os.path.dirname(benchmark_instances_file)

    current_network_name = ""
    n_verified = 0
    n_correct = 0
    total_time = 0.0
    network_index = 0
    for i, instance in enumerate(instances):
        print("Verifying instance number:", i)
        netname = os.path.realpath(os.path.join(instance_dir, instance[0]))
        # we work with pytorch nets, not onnx format
        netname = netname.replace(".onnx", ".pyt")
        if netname != current_network_name:
            print("Starting with network: ", netname)
            n_neurons_per_layer, n_layers = None, None
            try:
                network_size_info = netname.split("mnist-net_")[1].split(".pyt")[0]
                n_neurons_per_layer, n_layers = [
                    int(s) for s in network_size_info.split("x")
                ]
            except IndexError:
                pass
            original_network = load_net(netname, n_layers, n_neurons_per_layer)
            original_network.to(device)
            assert isinstance(original_network, nn.Sequential)
            network = AbstractNetwork.from_concrete_module(
                original_network, config.input_dim
            )
            freeze_network(network)

            verifier = MNBaBVerifier(
                network,
                device,
                make_verifier_config(**config),
            )
            current_network_name = netname
            network_index += 1
        if i < args.test_from:
            continue
        if args.test_num > 0 and i - args.test_from >= args.test_num:
            break
        vnn_lib_spec = os.path.join(instance_dir, instance[1])
        timeout_for_property = float(instance[2])

        input_constraints, output_constraints = parse_vnn_lib_prop(vnn_lib_spec)
        input_lb_arr, input_ub_arr = input_constraints[0]
        input_lb = torch.tensor(input_lb_arr).view(config.input_dim).to(device)
        input_ub = torch.tensor(input_ub_arr).view(config.input_dim).to(device)

        label = translate_constraints_to_label(output_constraints)[0]

        original_images, __ = translate_box_to_sample(
            input_constraints, equal_limits=True
        )
        original_image = (
            torch.tensor(original_images[0])
            .view(config.input_dim)
            .unsqueeze(0)
            .to(device)
        )
        pred_label = torch.argmax(original_network(original_image)).item()
        if pred_label != label:
            print("Network fails on test image, skipping.")
            continue
        else:
            n_correct += 1

        start_time = time.time()
        if verifier.verify(
            i, original_image, input_lb, input_ub, label, timeout_for_property
        ):
            n_verified += 1
            print("Verified instance: ", i)
        else:
            print("Unable to verify instance: ", i)
        instance_time = time.time() - start_time
        total_time += instance_time
        print("Iteration time: ", instance_time)
        print("Verified ", n_verified, " out of ", n_correct)
    print("Total time: ", total_time)
