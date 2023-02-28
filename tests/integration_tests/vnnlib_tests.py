import os

from bunch import Bunch  # type: ignore[import]

from src.utilities.argument_parsing import get_config_from_json
from src.utilities.loading.vnn_spec_loader import vnn_lib_data_loader
from src.verification_instance import VerificationInstance


def set_eran_config(cfg: Bunch) -> Bunch:
    config = Bunch()
    config.optimize_alpha = cfg.optimize_alpha
    config.alpha_lr = cfg.alpha_lr
    config.alpha_opt_iterations = 100
    config.optimize_prima = cfg.optimize_prima
    config.prima_lr = cfg.prima_lr
    config.prima_opt_iterations = cfg.prima_opt_iterations
    config.prima_hyperparameters = cfg.prima_hyperparameters
    config.peak_lr_scaling_factor = cfg.peak_lr_scaling_factor
    config.final_lr_div_factor = cfg.final_lr_div_factor
    config.beta_lr = cfg.beta_lr
    config.bab_batch_sizes = [2, 2, 8, 8, 12, 12, 20, 20, 32]
    config.branching = cfg.branching
    config.recompute_intermediate_bounds_after_branching = (
        cfg.recompute_intermediate_bounds_after_branching
    )
    config.use_dependence_sets = False
    config.use_early_termination = True
    return config


class TestVNNLib:
    def test_load_eran_specs(self) -> None:
        spec_path = "benchmarks_vnn21/eran/specs/mnist"
        data_loader_test, mean, std, is_nchw, input_shape = vnn_lib_data_loader(
            spec_path
        )

    def test_load_eran_instances(self) -> None:

        pref = "benchmarks_vnn21/eran/"
        instances_path = "eran_instances.csv"
        config_path = "configs/baseline/mnist_conv_big.json"
        config = get_config_from_json(config_path)
        # config = set_eran_config(config)
        verified_instances = []

        with open(pref + instances_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                net, spec, timeout = line.rstrip().split(",")
                if "mnist_relu" not in net:
                    continue
                onnx_path = os.path.join(pref, net)
                spec_path = os.path.join(pref, spec)

                v_inst = VerificationInstance.create_instance_from_vnn_spec(
                    onnx_path, spec_path, config
                )
                v_inst.run_instance()
                if v_inst.is_verified:
                    print("Verified")
                    verified_instances.append(spec_path)
                else:
                    print("Not verified")
        print("======== Succesfully verified instances ========")
        for inst in verified_instances:
            print(inst)

    def test_specific_instance(self) -> None:
        onnx_path = "benchmarks_vnn21/eran/nets/mnist_relu_9_200.onnx"
        spec_path = (
            "benchmarks_vnn21/eran/specs/mnist/mnist_spec_idx_4424_eps_0.01500.vnnlib"
        )
        config_path = "configs/baseline/mnist_conv_big.json"
        # spec_path = "benchmarks_vnn21/eran/specs/mnist/mnist_spec_idx_2697_eps_0.01500.vnnlib"
        config = get_config_from_json(config_path)
        config.timeout = 400
        v_inst = VerificationInstance.create_instance_from_vnn_spec(
            onnx_path, spec_path, config
        )
        v_inst.run_instance()
        if v_inst.is_verified:
            print("Verified")
        else:
            print("Not verified")


if __name__ == "__main__":
    t = TestVNNLib()
    t.test_specific_instance()
