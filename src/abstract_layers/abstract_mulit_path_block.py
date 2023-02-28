from __future__ import annotations

from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn

from src.abstract_domains.ai_util import AbstractElement
from src.abstract_layers.abstract_binary_op import BinaryOp
from src.abstract_layers.abstract_concat import Concat
from src.abstract_layers.abstract_container_module import (
    AbstractContainerModule,
    ActivationLayer,
)
from src.abstract_layers.abstract_sequential import Sequential
from src.abstract_layers.abstract_slice import Slice
from src.concrete_layers.binary_op import BinaryOp as concreteBinaryOp
from src.concrete_layers.concat import Concat as concreteConcat
from src.concrete_layers.multi_path_block import (
    MultiPathBlock as concreteMultiPathBlock,
)
from src.concrete_layers.slice import Slice as concreteSlice
from src.mn_bab_shape import AffineForm, MN_BaB_Shape
from src.state.subproblem_state import SubproblemState
from src.state.tags import LayerTag
from src.utilities.config import BacksubstitutionConfig


class MultiPathBlock(concreteMultiPathBlock, AbstractContainerModule):

    header: Optional[Slice]  # type: ignore[assignment]
    paths: nn.ModuleList  # type: ignore[assignment]
    merge: Union[Concat, BinaryOp]  # type: ignore[assignment]

    def __init__(
        self,
        header: Optional[concreteSlice],
        paths: List[nn.Sequential],
        merge: Union[concreteConcat, concreteBinaryOp],
        input_dim: Tuple[int, ...],
        **kwargs: Any,
    ) -> None:
        super(MultiPathBlock, self).__init__(header=header, paths=paths, merge=merge)

        # Header
        self.header: Optional[Slice] = None
        path_in_dim = input_dim
        if header is not None:
            assert isinstance(header, concreteSlice)
            self.header = Slice.from_concrete_module(header, input_dim, **kwargs)
            path_in_dim = self.header.output_dim

        # Paths
        abs_paths: List[Sequential] = []
        for path in paths:
            abs_paths.append(
                Sequential.from_concrete_module(path, path_in_dim, **kwargs)
            )
        self.paths = nn.ModuleList(abs_paths)

        # Merge
        merge_in_dims = [path.output_dim for path in abs_paths]  # TODO
        if isinstance(merge, concreteConcat):
            self.merge = Concat.from_concrete_module(merge, merge_in_dims, **kwargs)
        elif isinstance(merge, concreteBinaryOp):
            assert len(paths) == 2
            self.merge = BinaryOp.from_concrete_module(merge, merge_in_dims, **kwargs)
        else:
            assert False, f"Unknown merge block: {str(merge)}"

        # Other parameters
        self.output_dim = self.merge.output_dim
        self.bias = self.get_babsr_bias()

    @classmethod
    def from_concrete_module(  # type: ignore[override] # (checked at runtime)
        cls,
        module: concreteMultiPathBlock,
        input_dim: Tuple[int, ...],
        **kwargs: Any,
    ) -> MultiPathBlock:
        assert isinstance(module, concreteMultiPathBlock)
        abstract_layer = cls(  # Checked at runtime
            module.header,  # type: ignore[arg-type]
            module.paths,  # type: ignore[arg-type]
            module.merge,  # type: ignore[arg-type]
            input_dim,
            **kwargs,
        )
        return abstract_layer

    def backsubstitute_shape(
        self,
        config: BacksubstitutionConfig,
        input_lb: Tensor,
        input_ub: Tensor,
        abstract_shape: MN_BaB_Shape,
        from_layer_index: Optional[int],
        propagate_preceeding_callback: Optional[
            Callable[
                [BacksubstitutionConfig, MN_BaB_Shape, bool],
                Tuple[MN_BaB_Shape, Tuple[Tensor, Tensor]],
            ]
        ],  # = None
        preceeding_layers: Optional[List[Any]],  # = None
        use_early_termination_for_current_query: bool,  # = False
        full_back_prop: bool,  # = False
        optimize_intermediate_bounds: bool,  # = False
    ) -> Tuple[MN_BaB_Shape, Tuple[Tensor, Tensor]]:

        # Create corresponding preceeding callback
        propagate_preceeding_callback = self._get_header_callback(
            propagate_preceeding_callback
        )
        if preceeding_layers is not None:
            if self.header is not None:
                preceeding_layers = [*preceeding_layers, self.header]
        else:
            if self.header is not None:
                preceeding_layers = [self.header]

        orig_lb = abstract_shape.lb.clone()
        orig_ub: Optional[AffineForm] = None
        if abstract_shape.ub is not None:
            orig_ub = abstract_shape.ub.clone()

        unstable_queries_old_for_assert = abstract_shape.unstable_queries

        # Backprop through merge layer -> get individual shapes?
        pre_merge_shapes = self.merge.backsubstitute(config, abstract_shape)
        # Backprop through individual paths
        post_path_shapes: List[MN_BaB_Shape] = []
        for path_shape, path in zip(pre_merge_shapes, self.paths):
            (
                post_path_shape,
                (post_path_lbs, post_path_ubs),
            ) = path.backsubstitute_shape(
                config,
                input_lb,
                input_ub,
                path_shape,
                None,
                propagate_preceeding_callback,
                preceeding_layers,
                use_early_termination_for_current_query=False,
                full_back_prop=False,
                optimize_intermediate_bounds=optimize_intermediate_bounds,
            )
            post_path_shapes.append(post_path_shape)
            assert (
                abstract_shape.unstable_queries is None
                or (
                    abstract_shape.unstable_queries == unstable_queries_old_for_assert
                ).all()
            )

        # Backprop through header layer -> get one shape
        if self.header is not None:
            post_header_shape = self.header.backsubstitute(config, post_path_shapes)
        else:  # All paths are from the same input we can add them up
            final_lb_form = post_path_shapes[0].lb
            final_ub_form: Optional[AffineForm] = None
            if post_path_shapes[0].ub is not None:
                final_ub_form = post_path_shapes[0].ub

            for abs_shape in post_path_shapes[1:]:
                final_lb_form.coef += abs_shape.lb.coef
                final_lb_form.bias += abs_shape.lb.bias

                if abs_shape.ub is not None:
                    assert final_ub_form is not None
                    final_ub_form.coef += abs_shape.ub.coef
                    final_ub_form.bias += abs_shape.ub.bias

            post_header_shape = abstract_shape.clone_with_new_bounds(
                final_lb_form, final_ub_form
            )

        # Adjust bias
        new_lower: AffineForm
        new_upper: Optional[AffineForm] = None

        new_lb_bias = (
            post_header_shape.lb.bias - (len(self.paths) - 1) * orig_lb.bias
        )  # Both the shape in a and in b  contain the initial bias terms, so one has to be subtracted
        new_lb_coef = post_header_shape.lb.coef

        new_lower = AffineForm(new_lb_coef, new_lb_bias)

        if post_header_shape.ub is not None and orig_ub is not None:
            new_ub_bias = (
                post_header_shape.ub.bias - (len(self.paths) - 1) * orig_ub.bias
            )
            new_ub_coef = post_header_shape.ub.coef
            new_upper = AffineForm(new_ub_coef, new_ub_bias)

        abstract_shape.update_bounds(new_lower, new_upper)
        return (
            abstract_shape,
            (
                -np.inf * torch.ones_like(post_path_lbs, device=abstract_shape.device),
                np.inf * torch.ones_like(post_path_lbs, device=abstract_shape.device),
            ),
        )

    def get_babsr_bias(self) -> Tensor:
        biases: List[Tensor] = []
        is_cuda = False
        # In case one of them is one the gpu we will move both to gpu
        # Have to do this here as the paths (sequentials) are unaware of the device
        for p in self.paths:
            bias = p.get_babsr_bias()
            biases.append(bias.detach())
            if bias.is_cuda:
                is_cuda = True

        if is_cuda:
            c_biases: List[Tensor] = []
            for b in biases:
                c_biases.append(b.cuda())
            biases = c_biases

        bias_shape = biases[0].shape
        bias_numel = biases[0].numel()
        for b in biases:
            if b.numel() > bias_numel:
                bias_shape = b.shape

        base_sum: Tensor = biases[0].broadcast_to(bias_shape).clone()
        if len(biases) > 1:
            for b in biases[1:]:
                base_sum += b

        return nn.Parameter(base_sum)

    def reset_input_bounds(self) -> None:
        super(MultiPathBlock, self).reset_input_bounds()
        for path in self.paths:
            path.reset_input_bounds()

    def reset_optim_input_bounds(self) -> None:
        super(MultiPathBlock, self).reset_input_bounds()
        for path in self.paths:
            path.reset_optim_input_bounds()

    def reset_output_bounds(self) -> None:
        super(MultiPathBlock, self).reset_output_bounds()
        for path in self.paths:
            path.reset_output_bounds()

    def propagate_interval(
        self,
        interval: Tuple[Tensor, Tensor],
        use_existing_bounds: Optional[bool] = None,
        subproblem_state: Optional[SubproblemState] = None,
        activation_layer_only: bool = False,
        set_input: bool = True,
        set_output: bool = True,
    ) -> Tuple[Tensor, Tensor]:
        interval_list: List[Tuple[Tensor, Tensor]] = []
        if self.header is not None:
            interval_head = self.header.propagate_interval(
                interval,
                use_existing_bounds=use_existing_bounds,
                subproblem_state=subproblem_state,
                activation_layer_only=activation_layer_only,
                set_input=set_input,
                set_output=set_output,
            )
        else:
            interval_head = interval

        if isinstance(interval_head, List):
            assert len(interval_head) == len(self.paths)
            interval_list = interval_head
        else:
            interval_list = [interval_head for _ in self.paths]

        out_intervals: List[Tuple[Tensor, Tensor]] = []
        for input, path in zip(interval_list, self.paths):
            out_intervals.append(
                path.propagate_interval(
                    input,
                    use_existing_bounds=use_existing_bounds,
                    subproblem_state=subproblem_state,
                    activation_layer_only=activation_layer_only,
                    set_input=set_input,
                    set_output=set_output,
                )
            )

        merge_interval = self.merge.propagate_interval(
            out_intervals,
            use_existing_bounds=use_existing_bounds,
            subproblem_state=subproblem_state,
            activation_layer_only=activation_layer_only,
        )  # type ignore

        return merge_interval

    def propagate_abstract_element(
        self,
        abs_input: AbstractElement,
        use_existing_bounds: Optional[bool] = None,
        activation_layer_only: bool = False,
        set_input: bool = True,
        set_output: bool = True,
    ) -> AbstractElement:

        ae_list: List[AbstractElement] = []
        if self.header is not None:
            ae_head = self.header.propagate_abstract_element(
                abs_input,
                use_existing_bounds=use_existing_bounds,
                activation_layer_only=activation_layer_only,
                set_input=set_input,
                set_output=set_output,
            )
        else:
            ae_head = abs_input

        if isinstance(ae_head, List):
            assert len(ae_head) == len(self.paths)
            ae_list = ae_head
        else:
            ae_list = [ae_head for _ in self.paths]

        out_aes: List[AbstractElement] = []
        for path_input, path in zip(ae_list, self.paths):

            abs_output = path.propagate_abstract_element(
                path_input,
                use_existing_bounds,
                activation_layer_only,
                set_input=set_input,
                set_output=set_output,
            )
            out_aes.append(abs_output)

        out_ae = self.merge.propagate_abstract_element(
            out_aes,  # type: ignore [arg-type]
            use_existing_bounds=use_existing_bounds,
            activation_layer_only=activation_layer_only,
            set_input=set_input,
            set_output=set_output,
        )
        return out_ae

    def forward_pass(
        self,
        config: BacksubstitutionConfig,
        input_lb: Tensor,
        input_ub: Tensor,
        propagate_preceeding_callback: Optional[
            Callable[
                [BacksubstitutionConfig, MN_BaB_Shape, bool],
                Tuple[MN_BaB_Shape, Tuple[Tensor, Tensor]],
            ]
        ],
        preceeding_layers: Optional[List[Any]],
        ibp_call: Callable[[], None],
        timeout: float,
    ) -> None:
        header_callback = self._get_header_callback(propagate_preceeding_callback)
        for path in self.paths:
            path.forward_pass(
                config,
                input_lb,
                input_ub,
                header_callback,
                preceeding_layers,
                ibp_call,
                timeout,
            )

    def set_dependence_set_applicability(self, applicable: bool = True) -> None:
        is_applicable = True
        for path in self.paths:
            path.set_dependence_set_applicability(applicable)
            if path.layers[-1].dependence_set_applicable is not None:
                is_applicable &= path.layers[-1].dependence_set_applicable
            if not is_applicable:
                break
        self.dependence_set_applicable = is_applicable

    def get_default_split_constraints(
        self, batch_size: int, device: torch.device
    ) -> Dict[LayerTag, Tensor]:
        split_constraints: Dict[LayerTag, Tensor] = {}
        for path in self.paths:
            split_constraints.update(
                path.get_default_split_constraints(batch_size, device)
            )
        return split_constraints

    def get_default_split_points(
        self, batch_size: int, device: torch.device
    ) -> Dict[LayerTag, Tensor]:
        split_points: Dict[LayerTag, Tensor] = {}
        for path in self.paths:
            split_points.update(path.get_default_split_points(batch_size, device))
        return split_points

    def get_activation_layers(self) -> Dict[LayerTag, ActivationLayer]:
        act_layers: Dict[LayerTag, ActivationLayer] = {}
        for path in self.paths:
            act_layers.update(path.get_activation_layers())
        return act_layers

    def get_current_intermediate_bounds(
        self,
    ) -> OrderedDict[LayerTag, Tuple[Tensor, Tensor]]:
        intermediate_bounds: OrderedDict[
            LayerTag, Tuple[Tensor, Tensor]
        ] = OrderedDict()
        for path in self.paths:
            intermediate_bounds.update(path.get_current_intermediate_bounds())
        return intermediate_bounds

    def get_current_optimized_intermediate_bounds(
        self,
    ) -> OrderedDict[LayerTag, Tuple[Tensor, Tensor]]:
        intermediate_bounds: OrderedDict[
            LayerTag, Tuple[Tensor, Tensor]
        ] = OrderedDict()
        for path in self.paths:
            intermediate_bounds.update(path.get_current_optimized_intermediate_bounds())
        return intermediate_bounds

    def set_intermediate_input_bounds(
        self, intermediate_bounds: OrderedDict[LayerTag, Tuple[Tensor, Tensor]]
    ) -> None:
        for path in self.paths:
            path.set_intermediate_input_bounds(intermediate_bounds)

    def get_activation_layer_ids(
        self, act_layer_ids: Optional[List[LayerTag]] = None
    ) -> List[LayerTag]:
        if act_layer_ids is None:
            act_layer_ids = []
        for path in self.paths:
            act_layer_ids += path.get_activation_layer_ids()
        return act_layer_ids

    def get_relu_layer_ids(
        self, act_layer_ids: Optional[List[LayerTag]] = None
    ) -> List[LayerTag]:
        if act_layer_ids is None:
            act_layer_ids = []
        for path in self.paths:
            act_layer_ids += path.get_relu_layer_ids()
        return act_layer_ids

    def _get_header_callback(
        self,
        propagate_preceeding_callback: Optional[
            Callable[
                [BacksubstitutionConfig, MN_BaB_Shape, bool],
                Tuple[MN_BaB_Shape, Tuple[Tensor, Tensor]],
            ]
        ],
    ) -> Callable[
        [BacksubstitutionConfig, MN_BaB_Shape, bool],
        Tuple[MN_BaB_Shape, Tuple[Tensor, Tensor]],
    ]:
        """ReLU layers within the paths need a propagate preceeding callback that takes the header at the top into account"""

        def wrapped_call(
            config: BacksubstitutionConfig,
            abstract_shape: MN_BaB_Shape,
            use_early_termination_for_current_query: bool,
        ) -> Tuple[MN_BaB_Shape, Tuple[Tensor, Tensor]]:

            if self.header is not None:
                abstract_shape = self.header.backsubstitute(config, abstract_shape)

            if propagate_preceeding_callback is not None:
                return propagate_preceeding_callback(
                    config,
                    abstract_shape,
                    use_early_termination_for_current_query,
                )
            else:
                assert isinstance(abstract_shape.lb.coef, Tensor)
                bound_shape = abstract_shape.lb.coef.shape[:2]
                return (
                    abstract_shape,
                    (
                        -np.inf * torch.ones(bound_shape, device=abstract_shape.device),
                        np.inf * torch.ones(bound_shape, device=abstract_shape.device),
                    ),  # TODO: this seems unnecessary, move bounds into abstract_shape and just update them when it makes sense
                )

        return wrapped_call
