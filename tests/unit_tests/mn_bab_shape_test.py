import torch

from src.abstract_layers.abstract_relu import ReLU
from src.mn_bab_shape import AffineForm, MN_BaB_Shape
from src.state.tags import query_tag


class TestMNBaBShape:
    def test_init_with_full_argument_list(self) -> None:
        lb = AffineForm(torch.rand(1, 2, 2), torch.rand(1, 2, 2))
        ub = AffineForm(torch.rand(1, 2, 2), torch.rand(1, 2, 2))
        MN_BaB_Shape(
            query_id=query_tag(ReLU((1,))),
            query_prev_layer=None,
            queries_to_compute=None,
            lb=lb,
            ub=ub,
            unstable_queries=None,
            subproblem_state=None,
        )

    def test_init_with_partial_argument_list(self) -> None:
        lb = AffineForm(torch.rand(1, 2, 2))
        ub = AffineForm(torch.rand(1, 2, 2))
        MN_BaB_Shape(
            query_id=query_tag(ReLU((1,))),
            query_prev_layer=None,
            queries_to_compute=None,
            lb=lb,
            ub=ub,
            unstable_queries=None,
            subproblem_state=None,
        )
