import json
import os
from math import ceil, log10
from typing import Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42
rcParams["font.family"] = "serif"
rcParams["font.sans-serif"] = ["Palatio"]
rcParams["text.usetex"] = True

path_resnet6A_b_crown_babsr = "log_resnet6A_b-crown_babsr.json"
path_resnet6B_b_crown_babsr = "log_resnet6B_b-crown_babsr.json"

path_resnet6A_p4c_babsr = "log_resnet6A_p4c_babsr.json"
path_resnet6B_p4c_babsr = "log_resnet6B_p4c_babsr.json"

path_resnet6A_p4c_acs = "log_resnet6A_p4c_acs.json"
path_resnet6B_p4c_acs = "log_resnet6B_p4c_acs.json"

path_resnet6A_p4c_acs_cab = "log_resnet6A_p4c_acs+cab.json"
path_resnet6B_p4c_acs_cab = "log_resnet6B_p4c_acs+cab.json"


def build_full_path(path: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), path)


def get_verification_times(
    first_path: str, second_path: str
) -> Tuple[Sequence[float], Sequence[float]]:
    first_verification_times_log = json.load(open(build_full_path(first_path)))[
        "verfication_time"
    ]
    second_verification_times_log = json.load(open(build_full_path(second_path)))[
        "verfication_time"
    ]

    all_property_ids = sorted(
        list(
            set(
                list(first_verification_times_log.keys())
                + list(second_verification_times_log.keys())
            )
        )
    )

    first_verification_times = []
    second_verification_times = []
    for property_id in all_property_ids:
        if (
            (property_id in first_verification_times_log)
            and (property_id in second_verification_times_log)
            and first_verification_times_log[property_id] != float("inf")
            and second_verification_times_log[property_id] != float("inf")
        ):

            first_verification_times.append(first_verification_times_log[property_id])
            second_verification_times.append(second_verification_times_log[property_id])

    assert len(first_verification_times) == len(second_verification_times)
    return first_verification_times, second_verification_times


def _get_n_subproblems(
    first_path: str, second_path: str
) -> Tuple[Sequence[int], Sequence[int]]:
    first_n_subproblems_by_lower_bound = json.load(open(build_full_path(first_path)))[
        "n_subproblems_explored_to_reach_lower_bound"
    ]
    second_n_subproblems_by_lower_bound = json.load(open(build_full_path(second_path)))[
        "n_subproblems_explored_to_reach_lower_bound"
    ]

    first_n_subproblems = []
    second_n_subproblems = []

    for property_id in first_n_subproblems_by_lower_bound:
        if property_id in second_n_subproblems_by_lower_bound:
            first_n_subproblems_log_for_property = first_n_subproblems_by_lower_bound[
                property_id
            ]
            second_n_subproblems_log_for_property = second_n_subproblems_by_lower_bound[
                property_id
            ]
            if (
                not first_n_subproblems_log_for_property
                or not second_n_subproblems_log_for_property
            ):
                continue

            highest_common_lower_bound = min(
                first_n_subproblems_log_for_property[-1][0],
                second_n_subproblems_log_for_property[-1][0],
            )

            first_n_subproblems.append(
                _extract_n_subproblems_for(
                    highest_common_lower_bound, first_n_subproblems_log_for_property
                )
            )
            second_n_subproblems.append(
                _extract_n_subproblems_for(
                    highest_common_lower_bound, second_n_subproblems_log_for_property
                )
            )

    return first_n_subproblems, second_n_subproblems


def _extract_n_subproblems_for(
    lower_bound_threshold: float, n_subproblems_log: Sequence[Tuple[float, int]]
) -> int:
    for lb, n_subproblems in n_subproblems_log:
        if lb >= lower_bound_threshold:
            return n_subproblems
    raise ValueError("Expected log to contain lower bound.")


def plot_verification_times(
    first_verification_times_a: Sequence[float],
    first_verification_times_b: Sequence[float],
    second_verification_times_a: Sequence[float],
    second_verification_times_b: Sequence[float],
    first_label: str,
    second_label: str,
    plot_name: str,
) -> None:
    fontsize = 19  # adjust

    fig, ax = plt.subplots(figsize=(5, 4))
    # ax.set_aspect("auto")
    ax.set_aspect("equal", "box")

    # plt.xscale("log")
    plt.ylabel(second_label, rotation=0, fontsize=fontsize, ha="left")
    ax.yaxis.set_label_coords(-0.11, 1.05)
    plt.xlabel(first_label, fontsize=fontsize)
    plt.xlim([0, 450])
    # plt.xticks([1, 5, 10, 50], [1, 5, 10, 50])
    plt.ylim([0, 320])
    # plt.yticks(np.arange(0.8,1.01,0.2))
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))

    # ax.grid(which="major", axis="x", color=(1, 1, 1), zorder=0)
    # ax.grid(which="major", axis="y", color=(1, 1, 1), zorder=0)
    ax.set_facecolor((0.97, 0.97, 0.97))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    plt.setp(ax.get_xticklabels(), fontsize=fontsize)
    plt.setp(ax.get_yticklabels(), fontsize=fontsize)
    fig.tight_layout()
    plt.margins(0)

    ax.scatter(
        first_verification_times_a,
        second_verification_times_a,
        color="royalblue",
        alpha=0.8,
        s=20,
        zorder=3,
        label="ResNet6-A",
    )
    ax.scatter(
        first_verification_times_b,
        second_verification_times_b,
        color="orange",
        alpha=0.8,
        s=20,
        zorder=3,
        label="ResNet6-B",
    )

    ax.axline((1, 1), slope=1, color="black", alpha=0.3, ls="--")
    ax.legend(frameon=False, loc="upper left", fontsize=fontsize * 0.8)

    fig.savefig(plot_name, bbox_inches="tight")
    plt.clf()


def plot_subproblem_count_ratio(
    first_n_subproblems_a: Sequence[int],
    first_n_subproblems_b: Sequence[int],
    second_n_subproblems_a: Sequence[int],
    second_n_subproblems_b: Sequence[int],
    plot_name: str,
) -> None:
    fontsize = 19  # adjust

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.xaxis.set_major_locator(plt.MaxNLocator(2))
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))

    plt.ylabel(
        "Subproblem count ratio",
        rotation=0,
        fontsize=fontsize,
        ha="left",
    )
    ax.yaxis.set_label_coords(-0.11, 1.05)
    plt.xlabel("Quantile", fontsize=fontsize)

    # ax.grid(which="major", axis="x", color=(1, 1, 1), zorder=0)
    # ax.grid(which="major", axis="y", color=(1, 1, 1), zorder=0)
    ax.set_facecolor((0.97, 0.97, 0.97))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    plt.setp(ax.get_xticklabels(), fontsize=fontsize)
    plt.setp(ax.get_yticklabels(), fontsize=fontsize)
    plt.margins(0)
    fig.tight_layout()

    ratio_a = [
        float(first_n / second_n)
        for first_n, second_n in zip(first_n_subproblems_a, second_n_subproblems_a)
        if not ((first_n == 1) and (second_n == 1))
    ]
    mean_ratio_a = np.mean(ratio_a)
    quantiles_a = np.linspace(0, 1, len(ratio_a))
    ax.plot(
        quantiles_a,
        np.quantile(ratio_a, quantiles_a),
        color="royalblue",
        label="ResNet6-A",
    )
    ax.axhline(y=mean_ratio_a, color="royalblue", linestyle="--", alpha=0.8)
    ax.text(0.1, mean_ratio_a, "mean", verticalalignment="bottom", fontsize=14)

    ratio_b = [
        float(first_n / second_n)
        for first_n, second_n in zip(first_n_subproblems_b, second_n_subproblems_b)
        if not ((first_n == 1) and (second_n == 1))
    ]
    mean_ratio_b = np.mean(ratio_b)
    quantiles_b = np.linspace(0, 1, len(ratio_b))
    ax.axhline(y=mean_ratio_b, color="orange", linestyle="--", alpha=0.8)
    ax.plot(
        quantiles_b,
        np.quantile(ratio_b, quantiles_b),
        color="orange",
        label="ResNet6-B",
    )
    ax.text(0.4, mean_ratio_b, "mean", verticalalignment="bottom", fontsize=14)

    max_ratio = max(ratio_a + ratio_b)
    min_ratio = min(ratio_a + ratio_b)
    plt.ylim([min(1, min_ratio), 10 ** ceil(log10(max_ratio))])
    # ax.axhline(y=1, color="black", linestyle="--", alpha=0.2)
    ax.legend(frameon=False, loc="lower right", fontsize=fontsize * 0.85)

    plt.yscale("log")
    fig.savefig(plot_name, bbox_inches="tight")
    plt.clf()


if __name__ == "__main__":
    (
        resnet6A_b_crown_babsr_n_subproblems,
        resnet6A_p4c_babsr_n_subproblems,
    ) = _get_n_subproblems(path_resnet6A_b_crown_babsr, path_resnet6A_p4c_babsr)

    (
        resnet6B_b_crown_babsr_n_subproblems,
        resnet6B_p4c_babsr_n_subproblems,
    ) = _get_n_subproblems(path_resnet6B_b_crown_babsr, path_resnet6B_p4c_babsr)

    plot_subproblem_count_ratio(
        resnet6A_b_crown_babsr_n_subproblems,
        resnet6B_b_crown_babsr_n_subproblems,
        resnet6A_p4c_babsr_n_subproblems,
        resnet6B_p4c_babsr_n_subproblems,
        "ResNet6-AB_n_subproblems_ratio_w_wo_MNC.pdf",
    )

    (
        resnet6A_p4c_babsr_n_subproblems,
        resnet6A_p4c_acs_n_subproblems,
    ) = _get_n_subproblems(path_resnet6A_p4c_babsr, path_resnet6A_p4c_acs)
    (
        resnet6B_p4c_babsr_n_subproblems,
        resnet6B_p4c_acs_n_subproblems,
    ) = _get_n_subproblems(path_resnet6B_p4c_babsr, path_resnet6B_p4c_acs)

    plot_subproblem_count_ratio(
        resnet6A_p4c_babsr_n_subproblems,
        resnet6B_p4c_babsr_n_subproblems,
        resnet6A_p4c_acs_n_subproblems,
        resnet6B_p4c_acs_n_subproblems,
        "ResNet6-AB_n_subproblems_ratio_branching.pdf",
    )

    (
        resnet6A_p4c_acs_verification_times,
        resnet6A_p4c_acs_cab_verification_times,
    ) = get_verification_times(path_resnet6A_p4c_acs, path_resnet6A_p4c_acs_cab)
    (
        resnet6B_p4c_acs_verification_times,
        resnet6B_p4c_acs_cab_verification_times,
    ) = get_verification_times(path_resnet6B_p4c_acs, path_resnet6B_p4c_acs_cab)

    plot_verification_times(
        resnet6A_p4c_acs_verification_times,
        resnet6B_p4c_acs_verification_times,
        resnet6A_p4c_acs_cab_verification_times,
        resnet6B_p4c_acs_cab_verification_times,
        "Time [s] ACS",
        "Time [s] ACS+CAB",
        "ResNet6-AB_cab_runtime_comparison_p4c_acs.pdf",
    )

    (
        resnet6A_b_crown_babsr_verification_times,
        resnet6A_p4c_acs_cab_verification_times,
    ) = get_verification_times(path_resnet6A_b_crown_babsr, path_resnet6A_p4c_acs_cab)
    (
        resnet6B_b_crown_babsr_verification_times,
        resnet6B_p4c_acs_cab_verification_times,
    ) = get_verification_times(path_resnet6B_b_crown_babsr, path_resnet6B_p4c_acs_cab)

    plot_verification_times(
        resnet6A_b_crown_babsr_verification_times,
        resnet6B_b_crown_babsr_verification_times,
        resnet6A_p4c_acs_cab_verification_times,
        resnet6B_p4c_acs_cab_verification_times,
        "Time [s] wo. MNC BaBSR",
        "Time [s] MNC ACS+CAB",
        "ResNet6-AB_runtime_comparison_all.pdf",
    )
