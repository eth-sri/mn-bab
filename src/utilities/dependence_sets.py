from __future__ import annotations

from typing import List, Tuple

import numpy as np
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
        cstride: int = 1,
        cpadding: int = 0,
    ) -> None:
        self.sets = sets
        self.spatial_idxs = spatial_idxs  # Q indices in range [0, HW)
        self.cstride = cstride
        self.cpadding = cpadding
        self.device = sets.device
        self.batch_size = sets.shape[0]

    def clone(self) -> DependenceSets:
        return DependenceSets(
            self.sets.clone(),
            self.spatial_idxs.clone(),
            self.cstride,
            self.cpadding,
        )

    @staticmethod
    def unfold_to(x: Tensor, coef: DependenceSets) -> Tensor:
        """
        Extracts the entries of x (shape [B, 1?, c, h, w])
        corresponding to the dependence set of each neuron
        in coef.sets (shape [B, Q, c, d, d]), where (Q <= C * HW)
        The resulting shape is [B, Q, c, d, d].
        """
        if len(x.shape) == 5:
            x = x.squeeze(1)
        assert len(x.shape) == 4

        x_unfolded = F.unfold(
            x,
            kernel_size=coef.sets.shape[-1],
            stride=coef.cstride,
            padding=coef.cpadding,
        ).transpose(
            1, 2
        )  # [B, HW, c*d*d]

        x_unfolded = x_unfolded[:, coef.spatial_idxs, :]  # [B, Q, c*d*d]

        x_unfolded = x_unfolded.view(
            x_unfolded.shape[0], x_unfolded.shape[1], *coef.sets.shape[2:]
        )  # [B, Q, c, d, d]
        return x_unfolded

    def __add__(self, other: DependenceSets) -> DependenceSets:
        assert isinstance(other, DependenceSets), "Can only sum up two Dependence Sets"
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

        return type(self)(new_sets, self.spatial_idxs, self.cstride, new_padding)

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
