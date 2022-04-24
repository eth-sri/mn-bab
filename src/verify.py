import csv
import time

from comet_ml import Experiment

import torch

from src.abstract_layers.abstract_network import AbstractNetwork
from src.mn_bab_verifier import MNBaBVerifier
from src.utilities.argument_parsing import get_args, get_config_from_json
from src.utilities.initialization import seed_everthing
from src.utilities.loading.data import transform_and_bound
from src.utilities.loading.network import freeze_network, load_net_from

if __name__ == "__main__":
    args = get_args()
    config = get_config_from_json(args.config)
    seed_everthing(config.random_seed)

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

    original_network = load_net_from(config)
    original_network.to(device)
    original_network.eval()
    network = AbstractNetwork.from_concrete_module(original_network, config.input_dim)
    freeze_network(network)

    verifer = MNBaBVerifier(
        network,
        device,
        config.optimize_alpha,
        config.alpha_lr,
        config.alpha_opt_iterations,
        config.optimize_prima,
        config.prima_lr,
        config.prima_opt_iterations,
        config.prima_hyperparameters,
        config.peak_lr_scaling_factor,
        config.final_lr_div_factor,
        config.beta_lr,
        config.bab_batch_sizes,
        config.branching,
        config.recompute_intermediate_bounds_after_branching,
        config.use_dependence_sets,
        config.use_early_termination,
    )

    test_file = open(config.test_data_path, "r")
    test_instances = csv.reader(test_file, delimiter=",")

    total_start_time = time.time()
    running_total_time = 0.0
    n_verified = 0
    n_correct = 0
    for i, (label, *pixel_values) in enumerate(test_instances):
        if i < args.test_from:
            continue
        if args.test_num > 0 and i - args.test_from >= args.test_num:
            break
        start_time = time.time()
        print("Verifying instance number:", i)
        input, input_lb, input_ub = transform_and_bound(pixel_values, config, device)

        pred_label = torch.argmax(original_network(input)).item()
        if pred_label != int(label):
            print("Network fails on test image, skipping.")
            continue
        else:
            n_correct += 1

        if verifer.verify(i, input, input_lb, input_ub, int(label), config.timeout):
            n_verified += 1
            print("Verified instance: ", i)
        else:
            print("Unable to verify instance: ", i)
        iteration_time = time.time() - start_time
        running_total_time += iteration_time
        print("Iteration time: ", iteration_time)
        print("Running average verification time:", running_total_time / n_correct)
        print("Verified ", n_verified, " out of ", n_correct)
        print()
    if not experiment_logger.disabled:
        verifer.bab.log_info(experiment_logger)
    print("Total time: ", time.time() - total_start_time)
