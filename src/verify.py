import csv
import time
import sys
from comet_ml import Experiment  # type: ignore[import]

import torch
from torch import nn

from src.abstract_layers.abstract_network import AbstractNetwork
from src.mn_bab_verifier import MNBaBVerifier
from src.utilities.argument_parsing import get_args, get_config_from_json
from src.utilities.config import make_config, Dtype
from src.utilities.initialization import seed_everything
from src.utilities.loading.data import transform_and_bound
from src.utilities.loading.network import freeze_network, load_net, load_onnx_model
from src.utilities.logging import Logger, get_log_file_name
from src.verification_instance import VerificationInstance

if __name__ == "__main__":
    args = get_args()
    config_file = get_config_from_json(args.config)
    config = make_config(**config_file)
    seed_everything(config.random_seed)

    experiment_logger = Experiment(**config.logger.comet_options)
    experiment_logger.set_name(config.experiment_name)
    experiment_logger.log_parameters(config_file)

    logger = Logger(get_log_file_name(args.log_prefix), sys.stdout)
    sys.stdout = logger
    logger.log_default(config)

    if torch.cuda.is_available() and config.use_gpu:
        device = torch.device("cuda")
        experiment_logger.log_text("Using gpu")
        experiment_logger.log_text(torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        experiment_logger.log_text("Using cpu")

    net_format = config.network.path.split(".")[-1]
    if net_format in ["onnx", "gz"]:
        net_seq, onnx_shape, inp_name = load_onnx_model(config.network.path)  # Like this for mypy
        original_network: nn.Module = net_seq
        if len(config.input_dim) == 0:
            print(f"Setting shape: {onnx_shape}")
            config.input_dim = onnx_shape
    else:
        original_network = load_net(**config.network.load_params())
    original_network.to(device)
    original_network.eval()
    assert isinstance(original_network, nn.Sequential)

    if config.dtype == Dtype.float64:
        torch.set_default_dtype(torch.float64)
        original_network = original_network.double()
    else:
        torch.set_default_dtype(torch.float32)
        original_network = original_network.float()

    network = AbstractNetwork.from_concrete_module(
        original_network, config.input_dim
    ).to(device)
    freeze_network(network)
    num_classes = network.output_dim[-1]

    verifier = MNBaBVerifier(network, device, config.verifier)

    test_file = open(config.test_data_path, "r")
    test_instances = csv.reader(test_file, delimiter=",")

    total_start_time = time.time()
    running_total_time = 0.0
    n_correct = 0
    n_verified = 0
    n_disproved = 0
    for i, (label, *pixel_values) in enumerate(test_instances):
        label = int(label)
        network.reset_input_bounds()
        network.reset_output_bounds()
        network.reset_optim_input_bounds()
        if i < args.test_from:
            continue
        if args.test_num > 0 and i - args.test_from >= args.test_num:
            break
        input, input_lb, input_ub = transform_and_bound(pixel_values, config, device)

        pred_label = torch.argmax(original_network(input)).item()
        if pred_label != label:
            print("Network fails on test image, skipping.\n")
            continue
        else:
            n_correct += 1

        print("=" * 20)
        print("Verifying instance number:", i)

        inst = VerificationInstance.create_instance_for_batch_ver(network, verifier, input, input_lb, input_ub, label, config, num_classes)
        start_time = time.time()

        inst.run_instance()

        if inst.is_verified:
            n_verified += 1
            print("Verified instance: ", i)
        elif inst.adv_example is not None:
            n_disproved += 1
            print("Disproved instance: ", i)
        else:
            print("Unable to verify instance: ", i)

        inst.free_memory()

        iteration_time = time.time() - start_time
        running_total_time += iteration_time
        print("Iteration time: ", iteration_time)
        print("Running average verification time:", running_total_time / n_correct)
        print("Correct", n_correct, "out of", i+1)
        print("Verified", n_verified, "out of", n_correct-n_disproved)
        print("Disproved", n_disproved, "out of", n_correct - n_verified)
        print()
    if not experiment_logger.disabled:
        verifier.bab.log_info(experiment_logger)
    print("Total time: ", time.time() - total_start_time)
