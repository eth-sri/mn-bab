import argparse
import json

from bunch import Bunch


def get_args() -> argparse.Namespace:
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "-c",
        "--config",
        required=True,
        help="Add the Configuration file that has all the relevant parameters",
    )
    argparser.add_argument(
        "--test_from",
        default=0,
        type=int,
        help="First sample to evaluate",
    )
    argparser.add_argument(
        "--test_num",
        default=-1,
        type=int,
        help="Number of samples to evaluate",
    )
    return argparser.parse_args()


def get_config_from_json(json_file_path: str) -> Bunch:
    """
    Get the config from a json file and save it as a Bunch object.
    :param json_file:
    :return: config as Bunch object:
    """
    with open(json_file_path, "r") as config_file:
        config_dict = json.load(config_file)
    return Bunch(config_dict)
