import numpy as np

from src.utilities.prima_util import ActivationType, make_kactivation_obj


class TestPrimaConstraints:
    # example taken from: https://arxiv.org/pdf/2103.03638.pdf
    # test is very much brittle and can break, it is intended as
    # documentation of how the internal functions work
    def test_toy_example_in_paper(self) -> None:
        input_octahedron_constraints = np.array(
            [
                [2.0, 1.0, 1.0],
                [2.0, 1.0, 0.0],
                [2.0, 1.0, -1.0],
                [2.0, 0.0, 1.0],
                [2.0, 0.0, -1.0],
                [2.0, -1.0, 1.0],
                [2.0, -1.0, 0.0],
                [2.0, -1.0, -1.0],
            ]
        )
        constraint_object = make_kactivation_obj(ActivationType.ReLU)(
            input_octahedron_constraints
        )

        input_octahedron_constraints = np.array([[2.0, 1.0], [1.2, -1.0]])
        other_constraint_object = make_kactivation_obj(ActivationType.ReLU)(
            input_octahedron_constraints
        )

        """
        expected constraints:
        2    + x1 +         x2 - 2 * y1 - 2 * y2 >= 0
        0.75 +      0.375 * x2          -     y2 >= 0
             - x1              +     y1          >= 0
                  -         x2          +     y2 >= 0
                                     y1          >= 0
                                              y2 >= 0
        """
        assert np.allclose(
            np.array([1, 0.5, 0.5, -1, -1]), constraint_object.cons[0, :]
        )  # rescaled
        assert np.allclose(
            np.array([0.75, 0.375, -1]), other_constraint_object.cons[2, :]
        )
        assert np.allclose(
            np.array([0.0, -1.0, 0.0, 1.0, 0.0]), constraint_object.cons[5, :]
        )
        assert np.allclose(
            np.array([0.0, 0.0, -1.0, 0.0, 1.0]), constraint_object.cons[1, :]
        )
        assert np.allclose(
            np.array([0.0, 0.0, 0.0, 1.0, 0.0]), constraint_object.cons[3, :]
        )
        assert np.allclose(
            np.array([0.0, 0.0, 0.0, 0.0, 1.0]), constraint_object.cons[2, :]
        )
