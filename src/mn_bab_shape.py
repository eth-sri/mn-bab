from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, Union

import torch
from torch import Tensor

from src.state.tags import LayerTag, ParameterTag, QueryTag
from src.utilities.dependence_sets import DependenceSets
from src.utilities.queries import QueryCoef, filter_queries, num_queries
from src.verification_subproblem import SubproblemState

if TYPE_CHECKING:
    from src.abstract_layers.abstract_module import AbstractModule


class AffineForm:
    """
    An affine form mapping inputs to output bounds (may represent multiple queries)

    Attributes:
        coef: linear coefficient
        bias: bias (may be a scalar)
    """

    coef: QueryCoef  # batch_size x num_queries x current_layer_shape...
    bias: Tensor  # scalar or batch_size x num_queries

    batch_size: int
    num_queries: int
    device: torch.device

    def __init__(
        self,
        coef: Union[Tensor, DependenceSets],
        bias: Optional[Tensor] = None,  # TODO: pass device explicitly
    ):
        # assert bias is None or coef.device is bias.device # apparently this is actually not true
        self.device = coef.device
        self.batch_size = (
            coef.batch_size if isinstance(coef, DependenceSets) else coef.shape[0]
        )
        self.num_queries = num_queries(coef)
        self.coef = coef
        self.bias = (
            torch.zeros((self.batch_size, self.num_queries), device=coef.device)
            if bias is None
            else bias
        )
        assert self.bias.numel() == 1 or (
            self.bias.shape[0] == self.batch_size
            and self.bias.shape[1] == self.num_queries
        ), "{} {}".format(self.bias.numel(), self.bias.shape)

    def uses_dependence_sets(self) -> bool:
        return isinstance(self.coef, DependenceSets)

    def to(self, device: torch.device) -> AffineForm:
        coef = self.coef.to(device)
        bias = self.bias.to(device)
        return AffineForm(coef, bias)

    def clone(self) -> AffineForm:
        return AffineForm(self.coef.clone(), self.bias.clone())

    @property
    def is_leaf(self) -> bool:
        return self.coef.is_leaf and self.bias.is_leaf

    def detach(self) -> AffineForm:
        coef = self.coef.detach()
        bias = self.bias.detach()
        return AffineForm(coef, bias)

    def matches_filter_mask(self, mask: Tensor) -> bool:
        nb_entries = int(mask.sum())
        return nb_entries == self.num_queries

    def filter_queries(self, mask: Tensor) -> AffineForm:
        filtered_coef = filter_queries(self.coef, mask)
        filtered_bias = self.bias[:, mask] if self.bias.ndim > 0 else self.bias
        return AffineForm(filtered_coef, filtered_bias)


class MN_BaB_Shape:
    """
    An abstract shape that is defined by an affine lower and an
    optional upper bound on functions of the form
    f_i(x) = query_i*g(x). (where g is some part of a neural network)

    The abstract shape

    Attributes:
        query_id: queries that use the same query id share optimizable parameters
        query_prev_layer: optional. the previous layer for the query, needed to determine parameter sharing strategy
                          TODO: just store the parameter sharing strategy itself? (this is a bit more flexible and less ugly)

        num_queries: number of queries

        queries_to_compute: optional mask specifying which queries in the original layer we are bounding
                              - None means all
                              - uniform across batch
                            use cases:
                              - only recompute queries that are initially unstable

        lb: lower bound
        ub: upper bound (optional)

        unstable_queries: optional mask specifying which queries within the static mask are still active
                            - None means all
                            - uniform across batch
                          use cases:
                            - tracking unstable queries for early termination

        subproblem_state: the optimization state for all queries (optional)


        batch_size: every tensor has this batch dimension for parallelization (shape[0])
        device: every tensor is on this device

    """

    query_id: QueryTag  # queries that use the same query id share optimizable parameters
    query_prev_layer: Optional[AbstractModule]
    queries_to_compute: Optional[
        Tensor
    ]  # shape: total_num_queries_in_starting_layer, queries_to_compute.sum() == num_queries

    num_queries: int  # number of queries in each batch element

    lb: AffineForm
    ub: Optional[AffineForm]
    unstable_queries: Optional[Tensor]  # shape: num_queries

    subproblem_state: Optional[SubproblemState]

    batch_size: int
    device: torch.device

    def __init__(
        self,
        query_id: QueryTag,
        query_prev_layer: Optional[AbstractModule],
        queries_to_compute: Optional[Tensor],
        lb: AffineForm,
        ub: Optional[AffineForm],
        unstable_queries: Optional[Tensor],
        subproblem_state: Optional[SubproblemState],
    ) -> None:
        self.query_id = query_id
        self.query_prev_layer = query_prev_layer

        self.num_queries = lb.num_queries
        self.queries_to_compute = queries_to_compute

        if isinstance(lb.coef, DependenceSets):
            assert (
                subproblem_state is None
                or subproblem_state.constraints.prima_constraints is None
            )
        self.lb = lb.clone()

        if ub is not None:
            assert lb.device == ub.device
            assert lb.batch_size == ub.batch_size
            assert lb.num_queries == ub.num_queries
            self.ub = ub.clone()
        else:
            self.ub = None
        self.unstable_queries = unstable_queries

        self.subproblem_state = subproblem_state

        self.batch_size = lb.batch_size
        self.device = lb.device

        assert (
            subproblem_state is None or self.batch_size == subproblem_state.batch_size
        )

    # TODO @Timon Uncertain if we want this here but it is used in multiple files so it made more sense in the shape itself
    def clone_with_new_bounds(
        self, lb: AffineForm, ub: Optional[AffineForm]
    ) -> MN_BaB_Shape:
        return MN_BaB_Shape(
            query_id=self.query_id,
            query_prev_layer=self.query_prev_layer,
            queries_to_compute=self.queries_to_compute,
            lb=lb,
            ub=ub,
            unstable_queries=self.unstable_queries,
            subproblem_state=self.subproblem_state,
        )

    def uses_dependence_sets(self) -> bool:
        return self.lb.uses_dependence_sets()

    def initialize_unstable_queries(self) -> None:
        assert self.unstable_queries is None
        self.unstable_queries = torch.ones(
            self.num_queries,
            device=self.device,
            dtype=torch.bool,
        )

    def update_unstable_queries(self, current_unstable_queries: Tensor) -> None:
        assert self.unstable_queries is not None
        if (
            current_unstable_queries.all()
        ):  # TODO: does this actually pay off? (requires synchronization with the GPU)
            return  # if all current queries are unstable, no update is necessary
        new_unstable_queries = self.unstable_queries.clone()
        new_unstable_queries[self.unstable_queries] = current_unstable_queries
        filtered_lb = self.lb.filter_queries(current_unstable_queries)
        if self.ub is not None:
            filtered_ub = self.ub.filter_queries(current_unstable_queries)
        else:
            filtered_ub = None
        self.num_queries = filtered_lb.num_queries
        self.update_bounds(filtered_lb, filtered_ub)
        self.unstable_queries = new_unstable_queries
        assert self.matches_filter_mask(self.unstable_queries)

    def matches_filter_mask(self, mask: Tensor) -> bool:
        """
        Check whether the number of queries matches the number of entries
        in the filter.
        TODO: just move the entire filter logic into MN_BaB_Shape
        """
        nb_entries = int(mask.sum())
        return nb_entries == self.num_queries

    def get_unstable_queries_in_starting_layer(self) -> Optional[Tensor]:
        """
        Computes a mask of unstable queries relative to the starting layer.
        This information is currently tracked in two distinct tensors, therefore
        this have to combine queries_to_compute and unstable_queries.
        """
        if self.queries_to_compute is None:
            return self.unstable_queries
        if self.unstable_queries is None:
            return self.queries_to_compute
        query_filter_mask = self.queries_to_compute.clone()
        query_filter_mask[self.queries_to_compute] = self.unstable_queries
        return query_filter_mask

    @property
    def total_num_queries_in_starting_layer(self) -> int:
        if self.queries_to_compute is not None:
            assert len(self.queries_to_compute.shape) == 1
            return self.queries_to_compute.shape[0]
        if self.unstable_queries is not None:
            assert len(self.unstable_queries.shape) == 1
            return self.unstable_queries.shape[0]
        return self.num_queries

    def update_bounds(
        self,
        lb: AffineForm,
        ub: Optional[AffineForm],
    ) -> None:
        assert lb.device == self.device
        assert lb.batch_size == self.batch_size
        assert lb.num_queries == self.num_queries
        if ub is not None:  # TODO: Remove for performance again
            assert ub.device == self.device
            assert ub.batch_size == self.batch_size
            assert ub.num_queries == self.num_queries

            assert lb.coef is not ub.coef
            assert lb.bias is not ub.bias

        self.lb = lb  # TODO: this looks potentially dangerous
        self.ub = ub

    def concretize(
        self, input_lb: Tensor, input_ub: Tensor
    ) -> Tuple[Tensor, Optional[Tensor]]:

        output_lb, output_ub = self._matmul_of_coef_and_interval(
            input_lb.unsqueeze(1),  # add query dimension # had .expand(batch_repeats)
            input_ub.unsqueeze(1),  # add query dimension # had .expand(batch_repeats)
        )
        output_lb += self.lb.bias
        if output_ub is not None and self.ub is not None:  # For mypy
            output_ub += self.ub.bias

        assert len(output_lb.shape) == 2

        return output_lb, output_ub

    def get_input_corresponding_to_lower_bound(
        self, input_lb: Tensor, input_ub: Tensor
    ) -> Tensor:
        assert not self.uses_dependence_sets()
        assert isinstance(self.lb.coef, Tensor)
        return torch.where(
            self.lb.coef > 0, input_lb.unsqueeze(1), input_ub.unsqueeze(1)
        ).view(self.batch_size, self.lb.coef.shape[1], *input_lb.shape[1:])

    def get_parameters(
        self,
        parameter_key: ParameterTag,
        layer_id: LayerTag,
        make_default_parameters: Union[
            Callable[[torch.device], Tensor],
            Tuple[int, ...],  # default is a zero tensor
        ],
    ) -> Tensor:
        assert self.subproblem_state is not None
        return self.subproblem_state.parameters.get_parameters(
            self.query_id,
            parameter_key,
            layer_id,
            make_default_parameters,
        )

    def get_existing_parameters(
        self, parameter_key: ParameterTag, layer_id: LayerTag
    ) -> Tensor:
        assert self.subproblem_state is not None
        return self.subproblem_state.parameters.get_existing_parameters(
            self.query_id, parameter_key, layer_id
        )

    def change_alphas_to_WK_slopes(self) -> None:
        assert self.subproblem_state is not None
        layer_bounds = self.subproblem_state.constraints.layer_bounds
        return self.subproblem_state.parameters.change_alphas_to_WK_slopes(
            self.query_id, layer_bounds
        )

    def set_beta_parameters_to_zero(self) -> None:
        assert self.subproblem_state is not None
        return self.subproblem_state.parameters.set_beta_parameters_to_zero(
            self.query_id
        )

    def get_optimizable_parameters(  # TODO: get rid of this
        self, selected_query_id: Optional[QueryTag] = None, only_lb: bool = False
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]:
        assert self.subproblem_state is not None
        return self.subproblem_state.parameters.get_optimizable(
            selected_query_id=selected_query_id,
            only_lb=only_lb,
        )

    def get_split_constraints_for_relu(  # TODO: get rid of this
        self, layer_id: LayerTag, bounds: Tuple[Tensor, Tensor]
    ) -> Optional[Tensor]:
        if self.subproblem_state is None:
            return None
        if self.subproblem_state.constraints.split_state is None:
            return None
        # self.subproblem_state.constraints.split_state.refine_split_constraints_for_relu(
        #     layer_id, bounds
        # )
        return self.subproblem_state.constraints.split_state.split_constraints[layer_id]

    def get_split_constraints_for_sig(  # TODO: get rid of this
        self, layer_id: LayerTag, bounds: Tuple[Tensor, Tensor]
    ) -> Tuple[
        Optional[Tensor], Optional[Tensor]
    ]:  # for easy unpacking, could be Optional[Tuple[Tensor, Tensor]]
        if self.subproblem_state is None:
            return None, None
        if self.subproblem_state.constraints.split_state is None:
            return None, None
        # self.subproblem_state.constraints.split_state.refine_split_constraints_for_sig(layer_id, bounds)
        return (
            self.subproblem_state.constraints.split_state.split_constraints[layer_id],
            self.subproblem_state.constraints.split_state.split_points[layer_id],
        )

    def update_is_infeasible(self, is_infeasible: Tensor) -> None:
        assert self.subproblem_state is not None
        self.subproblem_state.constraints.update_is_infeasible(is_infeasible)

    def improve_layer_bounds(  # TODO: get rid of this
        self,
        new_intermediate_bounds: OrderedDict[
            LayerTag, Tuple[Tensor, Tensor]
        ],  # TODO: make this a LayerBounds as well?
    ) -> None:
        assert self.subproblem_state is not None
        self.subproblem_state.constraints.layer_bounds.improve(new_intermediate_bounds)

    def _elementwise_mul_of_coef_and_interval(
        self, interval_lb: Tensor, interval_ub: Tensor
    ) -> Tuple[Tensor, Optional[Tensor]]:

        ub_coef: Optional[Tensor] = None
        result_ub: Optional[Tensor] = None

        if self.uses_dependence_sets():
            assert isinstance(self.lb.coef, DependenceSets)
            interval_lb_matched = DependenceSets.unfold_to(interval_lb, self.lb.coef)
            interval_ub_matched = DependenceSets.unfold_to(
                interval_ub, self.lb.coef
            )  # NOTE Not optimal but we can assume that lb_coef and a potentially existing ub_coef have the same structure
            lb_coef = self.lb.coef.sets
            if self.ub is not None:
                assert isinstance(self.ub.coef, DependenceSets)
                ub_coef = self.ub.coef.sets
        else:
            assert isinstance(self.lb.coef, Tensor)
            interval_lb_matched = interval_lb
            interval_ub_matched = interval_ub
            assert len(interval_lb.shape) >= 3
            assert len(interval_ub.shape) >= 3
            lb_coef = self.lb.coef

            if self.ub is not None:
                assert isinstance(self.ub.coef, Tensor)
                ub_coef = self.ub.coef

        result_lb = lb_coef * torch.where(
            lb_coef >= 0,
            interval_lb_matched,
            interval_ub_matched,
        )

        if (
            self.ub is not None
            and interval_ub_matched is not None
            and ub_coef is not None
        ):
            result_ub = ub_coef * torch.where(
                ub_coef >= 0,
                interval_ub_matched,
                interval_lb_matched,
            )

        # result_jit_lb = self.clamp_mutiply(lb_coef, interval_lb_matched, interval_ub_matched)
        # result_jit_ub = self.clamp_mutiply(ub_coef, interval_ub_matched, interval_lb_matched)

        # assert torch.isclose(result_lb, result_jit_lb).all()
        # assert torch.isclose(result_ub, result_jit_ub).all()

        # result_idx_lb = self.index_mutiply(lb_coef, interval_lb_matched, interval_ub_matched)
        # result_idx_ub = self.index_mutiply(ub_coef, interval_ub_matched, interval_lb_matched)

        # assert torch.isclose(result_lb, result_idx_lb).all()
        # assert torch.isclose(result_ub, result_idx_ub).all()

        return result_lb, result_ub

    def _matmul_of_coef_and_interval(
        self, interval_lb: Tensor, interval_ub: Tensor
    ) -> Tuple[Tensor, Optional[Tensor]]:
        assert (
            len(interval_lb.shape) == len(interval_ub.shape) >= 3
        )  # batch_size x query_size x layer_size
        (
            elementwise_mul_lb,
            elementwise_mul_ub,
        ) = self._elementwise_mul_of_coef_and_interval(
            interval_lb,
            interval_ub,
        )

        num_query_dimensions = 2

        # LB
        if not elementwise_mul_lb.is_contiguous():
            elementwise_mul_lb = elementwise_mul_lb.contiguous()
        elementwise_mul_lb = elementwise_mul_lb.view(
            *elementwise_mul_lb.shape[:num_query_dimensions], -1
        )
        lb_mul_res = elementwise_mul_lb.sum(-1)
        # UB
        ub_mul_res: Optional[Tensor] = None
        if self.ub is not None:
            assert elementwise_mul_ub is not None
            if not elementwise_mul_ub.is_contiguous():
                elementwise_mul_ub = elementwise_mul_ub.contiguous()
            elementwise_mul_ub = elementwise_mul_ub.view(
                *elementwise_mul_ub.shape[:num_query_dimensions], -1
            )
            ub_mul_res = elementwise_mul_ub.sum(-1)

        return lb_mul_res, ub_mul_res

    @staticmethod
    @torch.jit.script
    def clamp_mutiply(
        A: Tensor, pos: Tensor, neg: Tensor
    ) -> Tensor:  # Produces worse gradients -also without
        Apos = A.clamp(min=0)
        Aneg = A.clamp(max=0)
        return pos * Apos + neg * Aneg

    @staticmethod
    @torch.jit.script
    def index_mutiply(A: Tensor, pos: Tensor, neg: Tensor) -> Tensor:
        result = A * torch.where(
            A >= 0,
            pos,
            neg,
        )
        return result
