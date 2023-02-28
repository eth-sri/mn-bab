from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


class DependenceSets:
    """
    A memory-efficient implementation of a coefficient matrix used in backsubstitution, as described in
    https://arxiv.org/abs/2007.10868. To enable set the `use_dependence_sets` flag.

    Enabling dependence sets optimizes intermediate backsubstitution queries that start after a convolutional layer.
    For query layer output size [C, H, W], the non-optimized coefficient matrix after backsubstituting through
    a layer with input size [c, h, w] ("current layer") has shape [B, CHW, c, h, w], where B is the batch size.
    With dependence sets this is reduced to [B, CHW, c, d, d] (`sets` attribute) where (d x d) is the spatial
    size of the dependence set of a single query neuron. The corresponding bias tensor is of shape [B, CHW], as usual. Concretizing an abstract shape with DependenceSets bounds produces a [B, CHW]
    shape, as usual.

    Additionally, cumulative stride (`cstride` attribute) and cumulative padding (`padding` attribute) are stored
    as a way to properly locate a dependence set of a query neuron within the current shape. The relationship is
    as follows: a 2D convolution with kernel size [c, d, d], stride cstride and padding cpadding, will visit the
    dependence sets of all HW query neurons in order (in any query channel, as the dependence sets don't change
    across C channels).

    This implementation assumes only Conv2D/ReLU layers before any Conv2D layer (no Normalization), symmetric stride
    and padding and dilation=groups=1 in all Conv2D layers, as well as square spatial dimensions of the input.
    """

    def __init__(
        self,
        sets: Tensor,
        spatial_idxs: Tensor,
        input_dim: Tuple[int, ...],
        cstride: int = 1,
        cpadding: int = 0,
    ) -> None:
        self.sets = sets
        self.spatial_idxs = spatial_idxs  # Q indices in range [0, HW)
        self.cstride = cstride
        self.cpadding = cpadding
        self.device = sets.device
        self.batch_size = sets.shape[0]
        self.input_dim = input_dim
        # assert sets.shape[-1] == sets.shape[-2]

    def to(self, device: torch.device) -> DependenceSets:
        return DependenceSets(
            self.sets.to(device),
            self.spatial_idxs.to(device),
            self.input_dim,
            self.cstride,
            self.cpadding,
        )

    def clone(self) -> DependenceSets:
        return DependenceSets(
            self.sets.clone(),
            self.spatial_idxs.clone(),
            self.input_dim,
            self.cstride,
            self.cpadding,
        )

    @property
    def is_leaf(self) -> bool:
        return self.sets.is_leaf and self.spatial_idxs.is_leaf

    def detach(self) -> DependenceSets:
        return DependenceSets(
            self.sets.detach(),
            self.spatial_idxs.detach(),
            self.input_dim,
            self.cstride,
            self.cpadding,
        )

    @staticmethod
    def unfold_to_spec(
        x: Tensor,
        padding: int,
        stride: int,
        kernel_size: int,
        input_dim: Tuple[int, ...],
        spatial_idxs: Tensor,
    ) -> Tensor:
        """
        Extracts the entries of x (shape [B, 1?, c, h, w])
        corresponding to the dependence set of each neuron
        in coef.sets (shape [B, Q, c, d, d]), where (Q <= C * HW)
        The resulting shape is [B, Q, c, d, d].
        """
        if len(x.shape) == 5:
            x = x.squeeze(1)
        assert len(x.shape) == 4

        _, _, h_x, w_x = x.shape
        h_unfolded = (h_x - kernel_size + 2 * padding) // stride + 1
        w_unfolded = (w_x - kernel_size + 2 * padding) // stride + 1

        input_c, input_h, input_w = input_dim
        idx_h = torch.div(spatial_idxs, input_w, rounding_mode="trunc")
        idx_w = spatial_idxs - idx_h * input_w

        x_unfolded = F.unfold(
            x,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        ).transpose(
            1, 2
        )  # [B, HW, c*d*d]

        x_unfolded = x_unfolded.view(x.shape[0], h_unfolded, w_unfolded, -1)[
            :, idx_h, idx_w, :
        ]  # [B, Q, c*d*d]

        x_unfolded = x_unfolded.view(
            x_unfolded.shape[0],
            x_unfolded.shape[1],
            x.shape[1],
            kernel_size,
            kernel_size,
        )  # [B, Q, c, d, d]
        return x_unfolded

    @staticmethod
    def _unfold_to_uniform(x: Tensor, coef: DependenceSets) -> Tensor:
        """
        Extracts the entries of x (shape [B, 1?, c, h, w])
        corresponding to the dependence set of each neuron
        in coef.sets (shape [B, Q, c, d, d]), where (Q <= C * HW)
        The resulting shape is [B, Q, c, d, d].
        """
        if len(x.shape) == 5:
            x = x.squeeze(1)
        assert len(x.shape) == 4
        _, num_queries, kernel_c, kernel_h, kernel_w = coef.sets.shape

        bs, _, h_x, w_x = x.shape
        h_unfolded = (h_x - kernel_h + 2 * coef.cpadding) // coef.cstride + 1
        w_unfolded = (w_x - kernel_w + 2 * coef.cpadding) // coef.cstride + 1

        input_c, input_h, input_w = coef.input_dim
        idx_h = torch.div(coef.spatial_idxs, input_w, rounding_mode="trunc")
        idx_w = coef.spatial_idxs - idx_h * input_w

        x_unfolded = F.unfold(
            x,
            kernel_size=(kernel_h, kernel_w),
            stride=coef.cstride,
            padding=coef.cpadding,
        ).transpose(
            1, 2
        )  # [B, HW, c*d*d]

        x_unfolded = x_unfolded.view(bs, h_unfolded, w_unfolded, -1)[
            :, idx_h, idx_w, :
        ]  # [B, Q, c*d*d]

        x_unfolded = x_unfolded.view(
            bs, num_queries, *coef.sets.shape[2:]
        )  # [B, Q, c, d, d]
        return x_unfolded

    def handle_padding(self, output_size: Tuple[int, ...]) -> None:
        device = self.device
        bs, num_queries, kernel_c, kernel_h, kernel_w = self.sets.shape
        output_c, output_h, output_w = output_size

        input_c, input_h, input_w = self.input_dim
        idx_h = torch.div(self.spatial_idxs, input_w, rounding_mode="trunc")
        idx_w = self.spatial_idxs - idx_h * input_w

        h_idx = (
            torch.arange(kernel_h, device=device).repeat(num_queries, 1)
            + idx_h.view(-1, 1).repeat(1, kernel_h) * self.cstride
        )  # torch.arange(kernel_h, device=device).repeat_interleave(num_queries) + idx_h.repeat(kernel_h) * self.cstride
        w_idx = (
            torch.arange(kernel_w, device=device).repeat(num_queries, 1)
            + idx_w.view(-1, 1).repeat(1, kernel_w) * self.cstride
        )

        mask = torch.ones(self.sets.shape[1], *self.sets.shape[3:], device=device)
        mask[h_idx < self.cpadding] = 0
        mask[h_idx >= output_h + self.cpadding] = 0
        mask = mask.permute(0, 2, 1)
        mask[w_idx < self.cpadding] = 0
        mask[w_idx >= output_w + self.cpadding] = 0
        mask = mask.permute(0, 2, 1)

        self.sets *= mask.unsqueeze(0).unsqueeze(2)

    @staticmethod
    def _unfold_to_non_uniform(x: Tensor, coef: DependenceSets) -> Tensor:
        """
        Extracts the entries of x (shape [B, Q, c, h, w])
        corresponding to the dependence set of each neuron
        in coef.sets (shape [B, Q, c, d, d]), where (Q <= C * HW)
        The resulting shape is [B, Q, c, d, d].
        """
        # TODO: does this do the right thing / is this an efficient implementation?

        assert (
            len(x.shape) == 5 and x.shape[1] != 1
        ), "uniform case should go to other implementation"
        _, num_queries, kernel_c, kernel_h, kernel_w = coef.sets.shape

        bs, _, _, h_x, w_x = x.shape
        h_unfolded = (h_x - kernel_h + 2 * coef.cpadding) // coef.cstride + 1
        w_unfolded = (w_x - kernel_w + 2 * coef.cpadding) // coef.cstride + 1

        input_c, input_h, input_w = coef.input_dim
        idx_h = coef.spatial_idxs // input_w
        idx_w = coef.spatial_idxs - idx_h * input_w

        x_unfolded = x.view(x.shape[0] * x.shape[1], *x.shape[2:])

        x_unfolded = F.unfold(
            x_unfolded,
            kernel_size=(kernel_h, kernel_w),
            stride=coef.cstride,
            padding=coef.cpadding,
        ).transpose(
            1, 2
        )  # [B*Q, HW, c*d*d]

        x_unfolded = x_unfolded.view(
            x.shape[0], x.shape[1], h_unfolded, w_unfolded, -1
        )  # [B, Q, HW, c*d*d]  # TODO: this seems huge. does this actually use space?

        # alternate implementation: issue is this probably creates a huge tensor here:
        # x_unfolded = x_unfolded[:, :, coef.spatial_idxs, :]  # [B, Q, Q, c*d*d]
        # x_unfolded = x_unfolded.diagonal(dim1=1, dim2=2)  # [B, c*d*d, Q]
        # x_unfolded = x_unfolded.transpose(dim0=1, dim1=2) # [B, Q, c*d*d]

        # manually index in parallel with python for comprehension instead:
        assert (
            x.shape[1] == coef.spatial_idxs.shape[0]
            and len(coef.spatial_idxs.shape) == 1
        )
        x_unfolded = torch.cat(
            tuple(
                x_unfolded[:, i, h, w].unsqueeze(1)
                for i, h, w in zip(range(x.shape[1]), idx_h, idx_w)
            ),
            dim=1,
        )  # [B, Q, c*d*d]

        x_unfolded = x_unfolded.view(
            x_unfolded.shape[0], x_unfolded.shape[1], *coef.sets.shape[2:]
        )  # [B, Q, c, d, d]
        return x_unfolded

    @staticmethod
    def unfold_to(x: Tensor, coef: DependenceSets) -> Tensor:
        """
        Extracts the entries of x (shape [B, (1|Q)?, c, h, w])
        corresponding to the dependence set of each neuron
        in coef.sets (shape [B, Q, c, d, d]), where (Q <= C * HW)
        The resulting shape is [B, Q, c, d, d].
        """
        if len(x.shape) == 4 or len(x.shape) == 5 and x.shape[1] == 1:
            return DependenceSets._unfold_to_uniform(x, coef)
        assert len(x.shape) == 5  # case [B, Q, c, h, w]
        return DependenceSets._unfold_to_non_uniform(x, coef)

    def __add__(self, other: DependenceSets) -> DependenceSets:
        assert isinstance(
            other, DependenceSets
        ), f"Can only sum up two Dependence Sets and not Dependence set and {type(other)}"
        assert (
            self.sets.shape[:3] == other.sets.shape[:3]
        ), "Can not sum Dependence Sets that disagree in channel or batch dimensions"
        assert (
            self.cstride == other.cstride
        ), "Can not sum Dependence Sets that disagree in stride"
        assert (
            self.spatial_idxs.shape == other.spatial_idxs.shape
            and (self.spatial_idxs == other.spatial_idxs).all()
        ), "Can not sum Dependence Sets that disagree in spatial indices"

        h_pad_self, h_pad_other = self.get_required_padding(
            self.sets.shape[-1], self.cpadding, other.sets.shape[-1], other.cpadding
        )
        v_pad_self, v_pad_other = self.get_required_padding(
            self.sets.shape[-2], self.cpadding, other.sets.shape[-2], other.cpadding
        )
        assert h_pad_self == v_pad_self
        assert h_pad_other == v_pad_other
        new_padding = self.cpadding + h_pad_self[0]

        new_sets = F.pad(self.sets, (*h_pad_self, *v_pad_self)) + F.pad(
            other.sets, (*h_pad_other, *v_pad_other)
        )

        return type(self)(
            new_sets, self.spatial_idxs, self.input_dim, self.cstride, new_padding
        )

    def __getitem__(self, item: Tuple[Union[int, slice], ...]) -> DependenceSets:
        assert all([x == slice(None, None, None) for x in item[2:]])
        return DependenceSets(
            self.sets[item[:2]],
            self.spatial_idxs[item[1]],
            self.input_dim,
            self.cstride,
            self.cpadding,
        )

    def to_tensor(self, output_size: Tuple[int, ...]) -> Tensor:
        bs, num_queries, kernel_c, kernel_h, kernel_w = self.sets.shape
        output_c, output_h, output_w = output_size
        input_c, input_h, input_w = self.input_dim

        assert kernel_h > self.cpadding
        assert kernel_w > self.cpadding
        assert output_h + 2 * self.cpadding >= kernel_h + (input_h - 1) * self.cstride
        assert output_w + 2 * self.cpadding >= kernel_w + (input_w - 1) * self.cstride

        n_kernel_neurons = kernel_c * kernel_h * kernel_w
        num_queries = self.spatial_idxs.shape[0]

        new_coef = torch.zeros(
            (
                *self.sets.shape[:2],
                output_c,
                output_h + 2 * self.cpadding,
                output_w + 2 * self.cpadding,
            ),
            device=self.device,
        )
        orig_stride: Tuple[int, ...] = new_coef.stride()
        new_coef = new_coef.flatten()

        strides_coef = [
            orig_stride[0],
            orig_stride[1],
            (output_h + 2 * self.cpadding) * self.cstride,
            self.cstride,
            orig_stride[2],
            output_w + self.cpadding + self.cpadding,
            1,
        ]
        strides_sets: Tuple[int, ...] = self.sets.stride()
        idx_h = torch.div(self.spatial_idxs, input_w, rounding_mode="trunc")
        idx_w = self.spatial_idxs - idx_h * input_w
        query_idxs = (
            torch.arange(num_queries, device=self.device)
            .repeat_interleave(bs)
            .repeat(n_kernel_neurons)
        )
        batch_idxs = (
            torch.arange(bs, device=self.device)
            .repeat(num_queries)
            .repeat(n_kernel_neurons)
        )
        h_idxs = idx_h.repeat_interleave(bs).repeat(n_kernel_neurons)
        w_idxs = idx_w.repeat_interleave(bs).repeat(n_kernel_neurons)
        c_idxs = (
            torch.arange(kernel_c, device=self.device)
            .repeat_interleave(bs * num_queries)
            .repeat(kernel_h * kernel_w)
        )
        kh_idxs = (
            torch.arange(kernel_h, device=self.device)
            .repeat_interleave(bs * num_queries * kernel_c)
            .repeat(kernel_w)
        )
        kw_idxs = torch.arange(kernel_w, device=self.device).repeat_interleave(
            bs * num_queries * kernel_c * kernel_h
        )
        idxs_coef = (
            batch_idxs * strides_coef[0]
            + query_idxs * strides_coef[1]
            + h_idxs * strides_coef[2]
            + w_idxs * strides_coef[3]
            + c_idxs * strides_coef[4]
            + kh_idxs * strides_coef[5]
            + kw_idxs * strides_coef[6]
        )
        idxs_sets = (
            batch_idxs * strides_sets[0]
            + query_idxs * strides_sets[1]
            + c_idxs * strides_sets[2]
            + kh_idxs * strides_sets[3]
            + kw_idxs * strides_sets[4]
        )
        new_coef[idxs_coef] = self.sets.flatten()[idxs_sets]
        new_coef = new_coef.view(
            (
                *self.sets.shape[:2],
                output_c,
                output_h + 2 * self.cpadding,
                output_w + 2 * self.cpadding,
            )
        )[
            :,
            :,
            :,
            self.cpadding : self.cpadding + output_h,
            self.cpadding : self.cpadding + output_w,
        ]

        # below implementation only works on cpu, but shows better what is done
        # new_coef_ = torch.zeros((*self.sets.shape[:2], output_c, output_h + 2 * self.cpadding, output_w + 2 * self.cpadding), device=self.device)
        # new_coef_strided = torch.as_strided(new_coef_,
        #                                     [bs, num_queries, input_h, input_w, output_c, kernel_h, kernel_w],
        #                                     [orig_stride[0], orig_stride[1], (output_h + 2 * self.cpadding) * self.cstride, self.cstride, orig_stride[2], output_w + self.cpadding + self.cpadding, 1]
        #                                     )  # type: ignore # Tuple index out of range?
        # query_idxs = torch.arange(num_queries, device=self.device)
        # new_coef_strided[:, query_idxs, idx_h, idx_w] = self.sets.contiguous()
        # new_coef_ = new_coef_[:, :, :, self.cpadding:self.cpadding+output_h, self.cpadding:self.cpadding+output_w]
        # assert torch.isclose(new_coef_,new_coef,atol=1e-10).all()
        return new_coef

    @classmethod
    def get_required_padding(
        cls, a_k: int, a_p: int, b_k: int, b_p: int
    ) -> Tuple[List[int], List[int]]:
        if a_k >= b_k:
            b_p_new = [int(np.ceil((a_k - b_k) / 2)), int(np.floor((a_k - b_k) / 2))]
            a_p_new = [0, 0]
            assert a_p == b_p_new[0] + b_p
            assert b_p_new[1] == b_p_new[0]
        else:
            b_p_new, a_p_new = cls.get_required_padding(b_k, b_p, a_k, a_p)

        return a_p_new, b_p_new
