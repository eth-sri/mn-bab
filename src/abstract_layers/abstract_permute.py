from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

from torch import Tensor

from src.abstract_domains.ai_util import AbstractElement
from src.abstract_layers.abstract_module import AbstractModule
from src.concrete_layers.permute import Permute as concretePermute
from src.mn_bab_shape import AffineForm, MN_BaB_Shape
from src.utilities.config import BacksubstitutionConfig
from src.verification_subproblem import SubproblemState

EPS = 1e-15


class Permute(concretePermute, AbstractModule):
    def __init__(self, perm_ind: Tuple[int, ...], input_dim: Tuple[int, ...]) -> None:
        """Generates the abstract permutation layer. Assumes that perm_ind contains an entry for the batch while input_dim does not. In case len(perm_ind) == len(input_dim) we cut the first dimension of the input_dim.

        Args:
            perm_ind (): _description_
            input_dim (Tuple[int, ...]): _description_
        """
        super(Permute, self).__init__(perm_ind)
        self.perm_ind = perm_ind
        if len(perm_ind) == len(input_dim):
            input_dim = input_dim[1:]
        self.input_dim = input_dim
        self.output_dim = tuple([input_dim[i - 1] for i in perm_ind[1:]])

        # As backsub queries have the dim query x neuron x input_dims
        # and perm_ind is batch x input dims we add a new index in front
        self.perm_ind = tuple([0] + [i + 1 for i in perm_ind])
        self.rev_perm_ind = [0] * len(self.perm_ind)
        for i in range(len(self.perm_ind)):
            self.rev_perm_ind[self.perm_ind[i]] = i

    @classmethod
    def from_concrete_module(  # type: ignore[override] # (checked at runtime)
        cls, module: concretePermute, input_dim: Tuple[int, ...], **kwargs: Any
    ) -> Permute:
        assert isinstance(module, concretePermute)
        return cls(module.dims, input_dim)

    def backsubstitute(
        self,
        config: BacksubstitutionConfig,
        abstract_shape: MN_BaB_Shape,
        intermediate_bounds_callback: Optional[
            Callable[[Tensor], Tuple[Tensor, Tensor]]
        ] = None,
    ) -> MN_BaB_Shape:
        new_lb_form = self._backsub_affine_form(abstract_shape.lb, abstract_shape)
        new_ub_form: Optional[AffineForm] = None
        if abstract_shape.ub is not None:
            new_ub_form = self._backsub_affine_form(abstract_shape.ub, abstract_shape)

        abstract_shape.update_bounds(new_lb_form, new_ub_form)
        return abstract_shape

    def _backsub_affine_form(
        self, affine_form: AffineForm, abstract_shape: MN_BaB_Shape
    ) -> AffineForm:
        assert isinstance(affine_form.coef, Tensor)
        new_bias = affine_form.bias
        new_coef = affine_form.coef.permute(self.rev_perm_ind)
        return AffineForm(new_coef, new_bias)

    def propagate_interval(
        self,
        interval: Tuple[Tensor, Tensor],
        use_existing_bounds: Optional[bool] = None,
        subproblem_state: Optional[SubproblemState] = None,
        activation_layer_only: bool = False,
        set_input: bool = True,
        set_output: bool = True,
    ) -> Tuple[Tensor, Tensor]:

        lb, ub = interval
        no_query_perm_ind = self.perm_ind
        if len(self.perm_ind) == len(lb.shape) + 1:
            no_query_perm_ind = tuple([i - 1 for i in self.perm_ind[1:]])
        return lb.permute(no_query_perm_ind), ub.permute(no_query_perm_ind)

    def propagate_abstract_element(
        self,
        abs_input: AbstractElement,
        use_existing_bounds: Optional[bool] = None,
        activation_layer_only: bool = False,
        set_input: bool = True,
        set_output: bool = True,
    ) -> AbstractElement:
        raise NotImplementedError
