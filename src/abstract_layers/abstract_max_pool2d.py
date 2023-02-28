from __future__ import annotations

from math import floor
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.abstract_domains.ai_util import AbstractElement
from src.abstract_layers.abstract_module import AbstractModule
from src.mn_bab_shape import AffineForm, MN_BaB_Shape
from src.utilities.config import BacksubstitutionConfig
from src.utilities.dependence_sets import DependenceSets
from src.verification_subproblem import SubproblemState


class MaxPool2d(nn.MaxPool2d, AbstractModule):
    kernel_size: Tuple[int, int]  # type: ignore[assignment] # hack
    stride: Tuple[int, int]  # type: ignore[assignment]
    padding: Tuple[int, int]  # type: ignore[assignment]
    dilation: Tuple[int, int]  # type: ignore[assignment]

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        input_dim: Tuple[int, ...],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
    ):

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        super(MaxPool2d, self).__init__(  # type: ignore # mypy issue 4335
            kernel_size, stride, padding, dilation
        )
        self.input_dim = input_dim
        output_height = floor(
            (
                input_dim[1]
                + 2 * self.padding[0]
                - self.dilation[0] * (self.kernel_size[0] - 1)
                - 1
            )
            / self.stride[0]
            + 1
        )
        output_width = floor(
            (
                input_dim[2]
                + 2 * self.padding[1]
                - self.dilation[1] * (self.kernel_size[1] - 1)
                - 1
            )
            / self.stride[1]
            + 1
        )
        self.output_dim = (input_dim[0], output_height, output_width)
        self.dependence_set_block = False

    @classmethod
    def from_concrete_module(  # type: ignore[override] # (checked at runtime)
        cls, module: nn.MaxPool2d, input_dim: Tuple[int, ...], **kwargs: Any
    ) -> MaxPool2d:
        assert isinstance(module, nn.MaxPool2d)
        abstract_layer = cls(
            module.kernel_size,
            input_dim,
            module.stride,
            module.padding,
            module.dilation,
        )
        return abstract_layer

    def backsubstitute(
        self,
        config: BacksubstitutionConfig,
        abstract_shape: MN_BaB_Shape,
        intermediate_bounds_callback: Optional[
            Callable[[Tensor], Tuple[Tensor, Tensor]]
        ] = None,
        prev_layer: Optional[AbstractModule] = None,
    ) -> MN_BaB_Shape:
        if self.input_bounds is None:
            raise RuntimeError("Cannot backsubstitute if bounds have not been set.")
        # if abstract_shape.uses_dependence_sets():
        #     assert False, "Dependence sets with MaxPool not implemented."
        # assert isinstance(abstract_shape.lb.coef, Tensor)

        input_lb = self.input_bounds[0].view(-1, *self.input_dim)
        input_ub = self.input_bounds[1].view(-1, *self.input_dim)

        # assert input_lb.shape[0] == 1

        output_ub = F.max_pool2d(input_lb, self.kernel_size, self.stride, self.padding)
        output_lb = F.max_pool2d(input_ub, self.kernel_size, self.stride, self.padding)
        output_dim = output_ub.shape
        input_dim = input_lb.shape

        pid_lb = F.pad(
            input_lb,
            (self.padding[1], self.padding[1], self.padding[0], self.padding[0]),
            value=-torch.inf,
        )
        pid_ub = F.pad(
            input_ub,
            (self.padding[1], self.padding[1], self.padding[0], self.padding[0]),
            value=-torch.inf,
        )

        if isinstance(abstract_shape.lb.coef, Tensor):
            tight = (output_ub == output_lb).all(0).all(0)
            lb_coef = abstract_shape.lb.coef.permute(1, 0, 2, 3, 4).flatten(start_dim=1)
            new_lb_coef = torch.zeros(
                (
                    abstract_shape.lb.num_queries,
                    abstract_shape.batch_size,
                    *self.input_dim,
                ),
                device=abstract_shape.device,
            ).flatten(start_dim=1)
            new_lb_bias = abstract_shape.lb.bias.clone().permute(1, 0)

            if abstract_shape.ub is not None:
                assert isinstance(abstract_shape.ub.coef, Tensor)
                ub_coef = abstract_shape.ub.coef.permute(1, 0, 2, 3, 4).flatten(
                    start_dim=1
                )
                new_ub_coef = torch.zeros(
                    (
                        abstract_shape.ub.num_queries,
                        abstract_shape.batch_size,
                        *self.input_dim,
                    ),
                    device=abstract_shape.device,
                ).flatten(start_dim=1)
                new_ub_bias = abstract_shape.ub.bias.clone().permute(1, 0)

            device = abstract_shape.device
            offsets_in = torch.tensor(
                [int(np.prod(input_dim[i + 1 :])) for i in range(len(input_dim))],
                device=device,
            )
            offsets_out = torch.tensor(
                [int(np.prod(output_dim[i + 1 :])) for i in range(len(output_dim))],
                device=device,
            )
            ch_range = torch.arange(output_dim[1], device=device).repeat(output_dim[0])
            bs_range = torch.arange(output_dim[0], device=device).repeat_interleave(
                output_dim[1]
            )

            new_lb_bias += (
                (
                    abstract_shape.lb.coef
                    * (output_lb * tight.unsqueeze(0).unsqueeze(0)).unsqueeze(
                        1
                    )  # add Batch, Query, Channel [H, W]
                )
                .flatten(2)
                .sum(2)
                .permute(1, 0)
            )

            if abstract_shape.ub is not None:
                new_ub_bias += (
                    (
                        abstract_shape.ub.coef
                        * (output_lb * tight.unsqueeze(0).unsqueeze(0)).unsqueeze(1)
                    )
                    .flatten(2)
                    .sum(2)
                    .permute(1, 0)
                )

            for y in torch.arange(output_dim[2])[~tight.all(1)]:
                for x in torch.arange(output_dim[3])[~tight[y]]:
                    if tight[y, x]:
                        assert False
                    # Get the input_window
                    w_in_idy = y * self.stride[0]
                    w_in_idx = x * self.stride[1]
                    w_lb = pid_lb[
                        :,
                        :,
                        w_in_idy : w_in_idy + self.kernel_size[0],
                        w_in_idx : w_in_idx + self.kernel_size[1],
                    ].flatten(start_dim=2)
                    w_ub = pid_ub[
                        :,
                        :,
                        w_in_idy : w_in_idy + self.kernel_size[0],
                        w_in_idx : w_in_idx + self.kernel_size[1],
                    ].flatten(start_dim=2)

                    best_lb, best_lb_i = w_lb.max(2)
                    max_ub = w_ub.max(2)[0]
                    strict_dom = (
                        torch.sum((best_lb.unsqueeze(2) <= w_ub).float(), 2) == 1.0
                    ).view(
                        -1
                    )  # Strict domination check

                    # Index of respective lower bound element (w.r.t. unpadded input window)
                    in_idx = (best_lb_i % self.kernel_size[1]).flatten()
                    in_idy = torch.div(
                        best_lb_i, self.kernel_size[1], rounding_mode="trunc"
                    ).flatten()
                    tot_idx = in_idx + w_in_idx - self.padding[0]
                    tot_idy = in_idy + w_in_idy - self.padding[1]

                    assert (
                        (tot_idx >= 0)
                        & (tot_idx < self.input_dim[2])
                        & (tot_idy >= 0)
                        & (tot_idy < self.input_dim[1])
                    ).all()

                    in_idx = (
                        bs_range * offsets_in[0]
                        + ch_range * offsets_in[1]
                        + tot_idy * offsets_in[2]
                        + tot_idx * offsets_in[3]
                    )
                    out_idx = (
                        bs_range * offsets_out[0]
                        + ch_range * offsets_out[1]
                        + y * offsets_out[2]
                        + x * offsets_out[3]
                    )

                    # Selected actual input
                    new_lb_coef[:, in_idx] += lb_coef[:, out_idx] * (
                        (lb_coef[:, out_idx] >= 0)
                        + (lb_coef[:, out_idx] < 0) * strict_dom
                    )

                    new_lb_bias += (
                        (
                            lb_coef[:, out_idx]
                            * (lb_coef[:, out_idx] < 0)
                            * (~strict_dom.unsqueeze(0))
                            * max_ub
                        )
                        .view(abstract_shape.num_queries, abstract_shape.batch_size, -1)
                        .sum(-1)
                    )

                    if abstract_shape.ub is not None:
                        new_ub_coef[:, in_idx] += ub_coef[:, out_idx] * (
                            (ub_coef[:, out_idx] < 0)
                            + (ub_coef[:, out_idx] >= 0) * strict_dom
                        )
                        new_ub_bias[:] += (
                            (
                                ub_coef[:, out_idx]
                                * (ub_coef[:, out_idx] >= 0)
                                * (~strict_dom)
                                * max_ub
                            )
                            .view(
                                abstract_shape.num_queries,
                                abstract_shape.batch_size,
                                -1,
                            )
                            .sum(-1)
                        )

            new_lb_coef = new_lb_coef.view(
                abstract_shape.lb.num_queries,
                abstract_shape.batch_size,
                *self.input_dim,
            ).permute(1, 0, 2, 3, 4)
            new_lb_form = AffineForm(new_lb_coef, new_lb_bias.permute(1, 0))

            # Upper bound
            if abstract_shape.ub is None:
                new_ub_form: Optional[AffineForm] = None
            else:
                new_ub_coef = new_ub_coef.view(
                    abstract_shape.lb.num_queries,
                    abstract_shape.batch_size,
                    *self.input_dim,
                ).permute(1, 0, 2, 3, 4)
                new_ub_form = AffineForm(new_ub_coef, new_ub_bias.permute(1, 0))
        elif isinstance(abstract_shape.lb.coef, DependenceSets):
            symmetric_stride = self.stride[0] == self.stride[1]
            symmetric_padding = self.padding[0] == self.padding[1]
            dilation_one = (
                self.dilation[0] == self.dilation[1] == 1 and self.dilation[0] == 1
            )
            dependence_sets_assumptions = (
                symmetric_stride and symmetric_padding and dilation_one
            )
            assert dependence_sets_assumptions

            (
                bs,
                num_queries,
                kernel_c,
                kernel_h,
                kernel_w,
            ) = abstract_shape.lb.coef.sets.shape

            c, input_h, input_w = abstract_shape.lb.coef.input_dim

            # Handle padding for DS, as unfold can only pad with 0's
            ds_padding = abstract_shape.lb.coef.cpadding * self.stride[0]
            ds_stride = abstract_shape.lb.coef.cstride * self.stride[0]
            kernel_size = (kernel_h - 1) * self.stride[0] + self.kernel_size[0]

            pid_lb = F.pad(
                pid_lb,
                (ds_padding,) * 4,
                value=-torch.inf,
            )
            pid_ub = F.pad(
                pid_ub,
                (ds_padding,) * 4,
                value=-torch.inf,
            )

            pid_lb = DependenceSets.unfold_to_spec(
                pid_lb,
                padding=0,
                stride=ds_stride,
                kernel_size=kernel_size,
                input_dim=(c, input_h, input_w),
                spatial_idxs=abstract_shape.lb.coef.spatial_idxs,
            ).flatten(end_dim=2)
            pid_ub = DependenceSets.unfold_to_spec(
                pid_ub,
                padding=0,
                stride=ds_stride,
                kernel_size=kernel_size,
                input_dim=(c, input_h, input_w),
                spatial_idxs=abstract_shape.lb.coef.spatial_idxs,
            ).flatten(end_dim=2)

            pod_lb = DependenceSets.unfold_to(
                output_lb,
                abstract_shape.lb.coef,
            ).flatten(end_dim=1)
            pod_ub = DependenceSets.unfold_to(
                output_ub,
                abstract_shape.lb.coef,
            ).flatten(end_dim=1)
            ds_padding += self.padding[0]
            tight = (pod_lb == pod_ub).all(0).all(0)

            # In contrast to Tensor MaxPool, Queries do not have the same dominance structure as they are in different positions
            lb_coef = abstract_shape.lb.coef.sets.flatten(
                end_dim=1
            )  # .flatten(start_dim=1) # Flatten to BS * Query * C x H * W
            new_lb_coef = torch.zeros(
                (
                    abstract_shape.batch_size,
                    abstract_shape.lb.num_queries,
                    kernel_c,
                    kernel_size,
                    kernel_size,
                ),
                device=abstract_shape.device,
            ).flatten(
                end_dim=2
            )  # .flatten(start_dim=1)
            new_lb_bias = abstract_shape.lb.bias.clone().flatten()

            if abstract_shape.ub is not None:
                assert isinstance(abstract_shape.ub.coef, DependenceSets)
                ub_coef = abstract_shape.ub.coef.sets.flatten(end_dim=1)
                new_ub_coef = (
                    torch.zeros(
                        (
                            abstract_shape.batch_size,
                            abstract_shape.lb.num_queries,
                            kernel_c,
                            kernel_size,
                            kernel_size,
                        ),
                        device=abstract_shape.device,
                    )
                    .flatten(end_dim=2)
                    .flatten()
                )
                new_ub_bias = abstract_shape.ub.bias.clone().flatten()

            # assert not tight.any()

            new_lb_bias += (
                (lb_coef * pod_lb * tight.unsqueeze(0).unsqueeze(0)).flatten(1).sum(1)
            )

            if abstract_shape.ub is not None:
                new_ub_bias += (
                    (ub_coef * pod_lb * tight.unsqueeze(0).unsqueeze(0))
                    .flatten(1)
                    .sum(1)
                )

            device = abstract_shape.device
            offsets_in = torch.tensor(new_lb_coef.stride(), device=device)
            new_lb_coef = new_lb_coef.flatten()
            offsets_out = torch.tensor(lb_coef.flatten(end_dim=1).stride())
            lb_coef = lb_coef.flatten()
            ub_coef = ub_coef.flatten()
            bsqc_range = torch.arange(bs * num_queries * kernel_c, device=device)

            for y in torch.arange(kernel_h)[~tight.all(1)]:
                for x in torch.arange(kernel_w)[~tight[y]]:
                    assert not tight[y, x]
                    # Get the input_window
                    w_in_idy = y * self.stride[0]
                    w_in_idx = x * self.stride[1]
                    w_lb = pid_lb[
                        :,
                        w_in_idy : w_in_idy + self.kernel_size[0],
                        w_in_idx : w_in_idx + self.kernel_size[1],
                    ].flatten(start_dim=1)
                    w_ub = pid_ub[
                        :,
                        w_in_idy : w_in_idy + self.kernel_size[0],
                        w_in_idx : w_in_idx + self.kernel_size[1],
                    ].flatten(start_dim=1)

                    best_lb, best_lb_i = w_lb.max(1)
                    max_ub = w_ub.max(1)[0]
                    strict_dom = (
                        torch.sum((best_lb.unsqueeze(1) <= w_ub).float(), 1) == 1.0
                    ).view(
                        -1
                    )  # Strict domination check

                    # Index of respective lower bound element (w.r.t. unpadded input window)
                    in_idx = (best_lb_i % self.kernel_size[1]).flatten()
                    in_idy = torch.div(
                        best_lb_i, self.kernel_size[1], rounding_mode="trunc"
                    ).flatten()
                    tot_idx = in_idx + w_in_idx  # - self.padding[0]
                    tot_idx[
                        tot_idx < 0
                    ] = 0  # Can happen for input blocks that are fully in the padding. Those will be cut away
                    tot_idy = in_idy + w_in_idy  # - self.padding[1]
                    tot_idy[tot_idy < 0] = 0

                    in_idx = (
                        bsqc_range * offsets_in[0]
                        + tot_idy * offsets_in[1]
                        + tot_idx * offsets_in[2]
                    )
                    out_idx = (
                        bsqc_range * offsets_out[0]
                        + y * offsets_out[1]
                        + x * offsets_out[2]
                    )

                    # Selected actual input
                    new_lb_coef[in_idx] += lb_coef[out_idx] * (
                        (lb_coef[out_idx] >= 0) + (lb_coef[out_idx] < 0) * strict_dom
                    )

                    new_lb_bias += (
                        (
                            lb_coef[out_idx]
                            * (lb_coef[out_idx] < 0)
                            * (~strict_dom.unsqueeze(0))
                            * max_ub
                        )
                        .view(bs * num_queries, -1)
                        .nan_to_num()
                        .sum(-1)
                    )

                    if abstract_shape.ub is not None:
                        new_ub_coef[in_idx] += ub_coef[out_idx] * (
                            (ub_coef[out_idx] < 0)
                            + (ub_coef[out_idx] >= 0) * strict_dom
                        )
                        new_ub_bias += (
                            (
                                ub_coef[out_idx]
                                * (ub_coef[out_idx] >= 0)
                                * (~strict_dom)
                                * max_ub
                            )
                            .view(bs * num_queries, -1)
                            .nan_to_num()
                            .sum(-1)
                        )

            new_lb_coef_ds = DependenceSets(
                new_lb_coef.view(
                    abstract_shape.batch_size,
                    num_queries,
                    kernel_c,
                    kernel_size,
                    kernel_size,
                ),
                abstract_shape.lb.coef.spatial_idxs,
                abstract_shape.lb.coef.input_dim,
                ds_stride,
                ds_padding,
            )
            new_lb_coef_ds.handle_padding(self.input_dim)

            new_lb_form = AffineForm(new_lb_coef_ds, new_lb_bias.view(bs, num_queries))

            # Upper bound
            if abstract_shape.ub is None:
                new_ub_form = None
            else:
                new_ub_coef_ds = DependenceSets(
                    new_ub_coef.view(
                        abstract_shape.batch_size,
                        num_queries,
                        kernel_c,
                        kernel_size,
                        kernel_size,
                    ),
                    abstract_shape.lb.coef.spatial_idxs,
                    abstract_shape.lb.coef.input_dim,
                    ds_stride,
                    ds_padding,
                )
                new_ub_coef_ds.handle_padding(self.input_dim)
                new_ub_form = AffineForm(
                    new_ub_coef_ds, new_ub_bias.view(bs, num_queries)
                )

        else:
            assert False

        abstract_shape.update_bounds(new_lb_form, new_ub_form)
        return abstract_shape

    def propagate_interval(
        self,
        interval: Tuple[Tensor, Tensor],
        use_existing_bounds: Optional[bool] = None,
        subproblem_state: Optional[SubproblemState] = None,
        activation_layer_only: bool = False,
        set_input: bool = True,
        set_output: bool = True,
    ) -> Tuple[Tensor, Tensor]:
        interval_lb, interval_ub = interval

        output_lb = F.max_pool2d(
            interval_lb, self.kernel_size, self.stride, self.padding, self.dilation
        )
        output_ub = F.max_pool2d(
            interval_ub, self.kernel_size, self.stride, self.padding, self.dilation
        )

        # assert (output_ub >= output_lb).all()

        return output_lb, output_ub

    def propagate_abstract_element(
        self,
        abs_input: AbstractElement,
        use_existing_bounds: Optional[bool] = None,
        activation_layer_only: bool = False,
        set_input: bool = True,
        set_output: bool = True,
    ) -> AbstractElement:
        return abs_input.max_pool2d(self.kernel_size, self.stride, self.padding)
