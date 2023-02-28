import csv
import time

from comet_ml import Experiment  # type: ignore[import]

import numpy as np
import pandas as pd  # type: ignore[import]
import torch
from torch import nn

from src.abstract_layers.abstract_network import AbstractNetwork
from src.mn_bab_verifier import MNBaBVerifier
from src.utilities.argument_parsing import get_args, get_config_from_json
from src.utilities.config import make_verifier_config
from src.utilities.initialization import seed_everything
from src.utilities.loading.data import transform_and_bound
from src.utilities.loading.network import freeze_network, load_net_from

CIFAR10_NORMALIZATION_STDDEV_COMPLETE_VERIFICATION = 0.225

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

    original_network = load_net_from(config)
    original_network.to(device)

    assert isinstance(original_network, nn.Sequential)
    network = AbstractNetwork.from_concrete_module(original_network, config.input_dim)
    freeze_network(network)

    verifer = MNBaBVerifier(
        network,
        device,
        make_verifier_config(**config),
    )

    test_file = open(config.test_data_path, "r")
    test_instances = list(csv.reader(test_file, delimiter=","))
    test_properties = pd.read_pickle(config.test_properties_path)
    time_to_solve = np.zeros(test_properties.shape[0])

    total_start_time = time.time()
    n_verified = 0
    for i, property in test_properties.iterrows():
        if i < args.test_from:
            continue
        if args.test_num > 0 and i - args.test_from >= args.test_num:
            break
        start_time = time.time()
        print("Verifying property number:", i)
        sample_index = int(property["Idx"])
        normalized_eps = property["Eps"]
        eps = normalized_eps * CIFAR10_NORMALIZATION_STDDEV_COMPLETE_VERIFICATION
        config.eps = eps
        competing_label = int(property["prop"])
        (label_str, *pixel_values) = test_instances[sample_index]
        label = int(label_str)
        input, input_lb, input_ub = transform_and_bound(pixel_values, config, device)

        pred_label = torch.argmax(original_network(input)).item()
        assert pred_label == label

        if verifer.verify_property(
            str(i), input_lb, input_ub, (label, competing_label, 0), config.timeout
        ):
            n_verified += 1
            print("Verified instance: ", i)
            time_to_solve[i] = time.time() - start_time
        else:
            print("Unable to verify instance: ", i)
            time_to_solve[i] = float("inf")
        print("Iteration time: ", time.time() - start_time)
        print("Verified ", n_verified, " out of ", i + 1)
    experiment_logger.log_table("time_per_instance.csv", time_to_solve)
    if not experiment_logger.disabled:
        verifer.bab.log_info(experiment_logger)
    print("Total time: ", time.time() - total_start_time)
