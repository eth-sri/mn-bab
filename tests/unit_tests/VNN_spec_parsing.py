import os
from typing import List, Tuple, Union

import numpy as np

from src.utilities.initialization import seed_everything
from src.utilities.loading.vnn_spec_loader import (
    parse_vnn_lib_prop,
    translate_constraints_to_label,
)


def is_float_try(str: str) -> bool:
    try:
        float(str)
        return True
    except ValueError:
        return False


def get_vnn_lib_files(dir: str) -> List[str]:
    file_list = []
    for path in os.listdir(dir):
        path = os.path.join(dir, path)
        if os.path.isdir(path):
            file_list += get_vnn_lib_files(path)
        elif path.endswith(".vnnlib"):
            file_list.append(path)
    return file_list


def gt_tuple_generator(n_class: int = 4) -> Tuple[int, int, float]:
    g3 = 0.0 if np.random.rand(0) > 0.3 else np.random.rand() * 5 - 2
    if g3 != 0.0:
        if np.random.rand() > 0.5:
            g1 = np.random.randint(0, n_class)
            g2 = -1
        else:
            g2 = np.random.randint(0, n_class)
            g1 = -1
    else:
        g1 = np.random.randint(0, n_class)
        g2 = np.random.randint(0, n_class)
    return g1, g2, g3


def input_box_generator(n_inputs: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    x = np.random.rand(n_inputs) * 2 - 1
    eps = np.random.rand(1) * 0.2
    return x + eps, x - eps


def write_vnn_spec(
    input_boxes: List[Tuple[np.ndarray, np.ndarray]],
    gt_tuples: Union[List[Tuple[int, int, float]], int],
    file_name: str,
    n_class: int,
) -> str:
    n_inputs = len(input_boxes[0][0])

    with open(file_name, "w") as f:
        f.write("\n; Definition of input variables\n")
        for i in range(n_inputs):
            f.write(f"(declare-const X_{i} Real)\n")

        f.write("\n; Definition of output variables\n")
        for i in range(n_class):
            f.write(f"(declare-const Y_{i} Real)\n")

        if isinstance(gt_tuples, int):
            x_ub, x_lb = input_boxes[0]
            f.write("\n; Definition of input constraints\n")
            for i in range(len(x_ub)):
                f.write(f"(assert (<= X_{i} {x_ub[i]:.8f}))\n")
                f.write(f"(assert (>= X_{i} {x_lb[i]:.8f}))\n")

            f.write("\n; Definition of output constraints\n")
            f.write("(assert (or\n")
            for i in range(n_class):
                if i == gt_tuples:
                    continue
                f.write(f"\t(and (>= Y_{i} Y_{gt_tuples}))\n")
            f.write("))\n")
        else:
            f.write("(assert (or\n")
            for i, (x_ub, x_lb) in enumerate(input_boxes):
                gt_tuple = gt_tuples[i]
                f.write("\t(and ")
                for i in range(len(x_ub)):
                    f.write(f"(<= X_{i} {x_ub[i]:.8f}) ")
                    f.write(f"(>= X_{i} {x_lb[i]:.8f}) ")
                if gt_tuple[0] == -1:
                    f.write(f"(>= Y_{gt_tuple[1]} {gt_tuple[2]})")
                elif gt_tuple[1] == -1:
                    f.write(f"(<= Y_{gt_tuple[0]} {gt_tuple[2]})")
                elif gt_tuple[2] == 0:
                    f.write(f"(>= Y_{gt_tuple[1]} Y_{gt_tuple[0]})")
                else:
                    f.write(f"(<= (Y_{gt_tuple[0]} - Y_{gt_tuple[1]}) {gt_tuple[2]})")
                f.write(")\n")
            f.write("))\n")
    return file_name


class TestSpec:
    """
    We test our ONNX parser implementation
    """

    def test_synthetic_gt(self, n: int = 5) -> None:
        seed_everything(42)
        n_class = 3
        n_input = 4

        for _ in range(n):
            gt_tuples = []
            input_boxes = []
            for _ in range(np.random.randint(1, 5, 1)[0]):
                gt_tuples.append(gt_tuple_generator(n_class))
                input_boxes.append(input_box_generator(n_input))

            file_name = os.path.join(os.path.dirname(__file__), "test.vnnlib")
            write_vnn_spec(input_boxes, gt_tuples, file_name, n_class)

            input_boxes_parsed, gt_tuples_parsed = parse_vnn_lib_prop(file_name)

            for i in range(len(input_boxes)):
                input_box = input_boxes_parsed[i]
                gt_tuple = gt_tuples_parsed[i]
                assert np.isclose(input_box[0], input_boxes[i][1]).all()
                assert np.isclose(input_box[1], input_boxes[i][0]).all()
                assert gt_tuple[0][0][0] == gt_tuples[i][0]
                assert gt_tuple[0][0][1] == gt_tuples[i][1]
                assert np.isclose(gt_tuple[0][0][2], gt_tuples[i][2])

    def test_synthetic_rob(self, n: int = 5) -> None:
        seed_everything(42)
        n_class = 3
        n_input = 4

        for _ in range(n):
            gt_tuples = np.random.randint(0, n_class)
            input_boxes = [input_box_generator(n_input)]

            file_name = os.path.join(os.path.dirname(__file__), "test.vnnlib")
            write_vnn_spec(input_boxes, gt_tuples, file_name, n_class)

            input_boxes, output_gt_constraints = parse_vnn_lib_prop(file_name)
            y = translate_constraints_to_label(output_gt_constraints)[0]
            assert gt_tuples == y

    def test_nn4sys(self) -> None:
        input_boxes, output_gt_constraints = parse_vnn_lib_prop(
            "benchmarks_vnn21/nn4sys/specs/lognormal_100_13_14_15_19_20.vnnlib"
        )
        input_boxes, output_gt_constraints = parse_vnn_lib_prop(
            "benchmarks_vnn21/nn4sys/specs/lognormal_100_1.vnnlib"
        )
        input_boxes, output_gt_constraints = parse_vnn_lib_prop(
            "benchmarks_vnn21/nn4sys/specs/lognormal_1000_1_3_4_6_7_8_9_10_11_12_13_14_15_16_18_19_21_22_24_26_28_30_32_34_36_37_39_40_41_43_44_46_47_49_50_51_53_54_55_57_58_59_61_62_63_64_66_67_68_71.vnnlib"
        )

    # def test_all_old(self) -> None:
    #     vnn_lib_files = get_vnn_lib_files("benchmarks_vnn21")
    #     for i, vnn_lib_file in enumerate(vnn_lib_files):
    #         input_boxes, output_gt_constraints = parse_vnn_lib_prop(vnn_lib_file)
    #
    # def test_all_new(self) -> None:
    #     parent_dir = "vnn-comp-2022-sup/benchmark_vnn22"
    #     exception_list = ["nn4sys2022", "colins_aerospace"]
    #     for path in os.listdir(parent_dir):
    #         if path in exception_list:
    #             print(f"Benchmark {path} skipped")
    #             continue
    #         benchmark_dir = os.path.join(parent_dir, path)
    #         vnn_lib_files = get_vnn_lib_files(benchmark_dir)
    #         try:
    #             for i, vnn_lib_file in enumerate(vnn_lib_files):
    #                 input_boxes, output_gt_constraints = parse_vnn_lib_prop(vnn_lib_file)
    #             print(f"Successfully parsed all files in {path}")
    #         except Exception as e:
    #             print(f"Failed to parse all files in {path} at file {vnn_lib_file}")
    #             raise e


if __name__ == "__main__":
    T = TestSpec()
    # T.test_nn4sys()
    # T.test_all_new()
    # T.test_all_old()
    T.test_synthetic_gt(50)
    T.test_synthetic_rob(50)
