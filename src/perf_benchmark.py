import argparse
import pstats
from cProfile import Profile

from torch.profiler import ProfilerActivity, profile, record_function

from src.verification_instance import VerificationInstance, create_instances_from_args


def run_torch_benchmark_on_instance(instance: VerificationInstance) -> None:

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        with_stack=True,
        record_shapes=True,
    ) as prof:
        with record_function("instance_perf_bench"):
            instance.run_instance()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    print(
        prof.key_averages(group_by_input_shape=True).table(
            sort_by="cuda_time_total", row_limit=20
        )
    )
    print(
        prof.key_averages(group_by_stack_n=5).table(
            sort_by="self_cuda_time_total", row_limit=10
        )
    )

    prof.export_chrome_trace("perf_bench.json")
    prof.export_stacks("./perf_bench_stacks.txt", "self_cuda_time_total")


def run_cperf_benchmark_on_instance(instance: VerificationInstance) -> None:

    profiler = Profile()
    profiler.enable()
    instance.run_instance()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("tottime")

    # Print the stats report
    stats.strip_dirs()
    stats.sort_stats(pstats.SortKey.CUMULATIVE)
    stats.reverse_order()
    stats.print_stats()
    stats.dump_stats("cperf.perf")
    stats.sort_stats(pstats.SortKey.TIME)
    stats.reverse_order()
    stats.print_stats()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run verification instances on the vnn21 (& vnn22) datasets. Simply provide the corresponding nets, specs, and configs"
    )
    parser.add_argument(
        "-c",
        "--configs",
        type=str,
        nargs="*",
        help="The configs corresponding to the nets x specs. Either we load a single config for all specs or one config for each spec",
    )
    args = parser.parse_args()
    args.instances = None
    args.nets = None

    instances = create_instances_from_args(args)
    instance_list = list(instances.values())
    instance = instance_list[0][0]
    instance.config.timeout = 120
    run_cperf_benchmark_on_instance(instance)
