# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module, Parameter

from fairseq2.nn.utils.mask import compute_mask


class Wav2Vec2Mask(Module):
    """Masks feature extractor outputs as described in Section 3.1 of
    :cite:t:`baevski2020wav2vec`."""

    mask_len: int
    mask_prob: float
    mask_embed: Parameter
    spatial_mask_len: int
    spatial_mask_prob: float

    def __init__(
        self,
        model_dim: int,
        mask_len: int = 10,
        mask_prob: float = 0.65,
        spatial_mask_len: int = 10,
        spatial_mask_prob: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """
        :param model_dim:
            The dimensionality of the model.
        :param mask_len:
            The length of the temporal mask that is applied over time steps.
        :param mask_prob:
            The probability of masking a time step.
        :param spatial_mask_len:
            The length of the spatial mask that is applied over features.
        :param spatial_mask_prob:
            The probability of masking a feature.
        """
        super().__init__()

        self.mask_len = mask_len
        self.mask_prob = mask_prob

        self.mask_embed = Parameter(
            torch.empty((model_dim,), device=device, dtype=dtype)
        )

        self.spatial_mask_len = spatial_mask_len
        self.spatial_mask_prob = spatial_mask_prob

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the parameters of the module."""
        nn.init.uniform_(self.mask_embed)

    def forward(
        self, x: Tensor, seq_lens: Optional[Tensor]
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        :param x:
            The input to mask. *Shape:* :math:`(N,S,M)`, or :math:`(S,M)` when
            unbatched, where :math:`N` is the batch size, :math:`S` is the
            sequence length, and :math:`M` is the dimensionality of the model.
        :param seq_lens:
            An array where each element represents the length of the sequence at
            the same index in ``x``. *Shape:* :math:`(N)`, :math:`(N,1)`, or
            :math:`()` when unbatched, where :math:`N` is the batch size.

        :returns:
            - ``x`` with mask applied. *Shape:* Same as ``x``.
            - The boolean temporal mask that has been applied to ``x``. *Shape:*
              :math:`(N,S)` or :math:`(S)` when unbatched, where :math:`N` is
              the batch size and :math`S` is the sequence length.

        .. note::
            For a boolean mask, a ``True`` indicates that the corresponding
            position should be masked.
        """
        if not self.training:
            return x, None

        seq_len, model_dim = x.shape[-2:]

        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.size(0)

        # Temporal mask over time steps.
        if self.mask_prob > 0.0:
            shape = (batch_size, seq_len)

            mask = compute_mask(
                shape,
                self.mask_len,
                self.mask_prob,
                seq_lens,
                min_num_masks=2,
                device=x.device,
            )

            if mask is not None:
                if x.dim() == 2:
                    mask = mask.squeeze(0)

                x[mask] = self.mask_embed
        else:
            mask = None

        # Spatial mask over features.
        if self.spatial_mask_prob > 0.0:
            shape = (batch_size, model_dim)

            spatial_mask = compute_mask(
                shape,
                self.spatial_mask_len,
                self.spatial_mask_prob,
                min_num_masks=2,
                device=x.device,
            )

            if spatial_mask is not None:
                spatial_mask = spatial_mask.unsqueeze(1).expand(-1, seq_len, -1)

                if x.dim() == 2:
                    spatial_mask = spatial_mask.squeeze(0)

                x[spatial_mask] = 0.0

        return x, mask

    def extra_repr(self) -> str:
        """:meta private:"""
        return f"mask_len={self.mask_len}, mask_prob={self.mask_prob}, spatial_mask_len={self.spatial_mask_len}, spatial_mask_prob={self.spatial_mask_prob}"
