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

from fairseq2.nn.padding import PaddingMask
from fairseq2.nn.utils.mask import compute_row_mask
from fairseq2.typing import DataType, Device


class Wav2Vec2Masker(Module):
    """Masks extracted features as described in Section 3.1 of
    :cite:t:`https://doi.org/10.48550/arxiv.2006.11477`."""

    temporal_span_len: int
    max_temporal_mask_prob: float
    temporal_mask_embed: Parameter
    spatial_span_len: int
    max_spatial_mask_prob: float

    def __init__(
        self,
        model_dim: int,
        temporal_span_len: int = 10,
        max_temporal_mask_prob: float = 0.65,
        spatial_span_len: int = 10,
        max_spatial_mask_prob: float = 0.0,
        *,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param model_dim:
            The dimensionality of the model.
        :param temporal_span_len:
            The length of each temporal mask span that is applied over time
            steps.
        :param max_temporal_mask_prob:
            The maximum probability of masking a time step. Note that, due to
            mask span overlap, the effective probability might be smaller.
        :param spatial_span_len:
            The length of each spatial mask span that is applied over features.
        :param max_spatial_mask_prob:
            The maximum probability of masking a feature. Note that, due to mask
            span overlap, the effective probability might be smaller.
        """
        super().__init__()

        if max_temporal_mask_prob == 0.0:
            raise ValueError("`max_temporal_mask_prob` must be greater than 0.")

        self.temporal_span_len = temporal_span_len
        self.max_temporal_mask_prob = max_temporal_mask_prob

        self.temporal_mask_embed = Parameter(
            torch.empty((model_dim,), device=device, dtype=dtype)
        )

        self.spatial_span_len = spatial_span_len
        self.max_spatial_mask_prob = max_spatial_mask_prob

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the parameters and buffers of the module."""
        nn.init.uniform_(self.temporal_mask_embed)

    def forward(
        self, seqs: Tensor, padding_mask: Optional[PaddingMask]
    ) -> Tuple[Tensor, Tensor]:
        """
        :param seqs:
            The sequences to mask. *Shape:* :math:`(N,S,M)`, where :math:`N` is
            the batch size, :math:`S` is the sequence length, and :math:`M` is
            the dimensionality of the model.
        :param seq_lens:
            An array where each element represents the length of the sequence at
            the same index in ``seqs``. *Shape:* :math:`(N)`, where :math:`N` is
            the batch size.

        :returns:
            - The input sequences with mask applied. *Shape:* Same as ``seqs``.
            - The temporal mask that has been applied to ``seqs``. *Shape:*
              :math:`(N,S)`, where :math:`N` is the batch size and :math`S` is
              the sequence length.
        """
        batch_size, seq_len, model_dim = seqs.shape

        # Temporal mask over time steps.
        temporal_mask = compute_row_mask(
            shape=(batch_size, seq_len),
            span_len=self.temporal_span_len,
            max_mask_prob=self.max_temporal_mask_prob,
            row_lens=padding_mask.seq_lens if padding_mask is not None else None,
            min_num_spans=2,
            device=seqs.device,
        )

        assert temporal_mask is not None

        seqs[temporal_mask] = self.temporal_mask_embed

        if self.max_spatial_mask_prob > 0.0:
            # Spatial mask over features.
            # (N, M)
            spatial_mask = compute_row_mask(
                shape=(batch_size, model_dim),
                span_len=self.spatial_span_len,
                max_mask_prob=self.max_spatial_mask_prob,
                min_num_spans=2,
                device=seqs.device,
            )

            assert spatial_mask is not None

            # (N, M) -> (N, S, M)
            spatial_mask = spatial_mask.unsqueeze(1).expand(-1, seq_len, -1)

            seqs[spatial_mask] = 0.0

        return seqs, temporal_mask

    def extra_repr(self) -> str:
        """:meta private:"""
        return (
            f"temporal_span_len={self.temporal_span_len}, "
            f"max_temporal_mask_prob={self.max_temporal_mask_prob}, "
            f"spatial_span_len={self.spatial_span_len}, "
            f"max_spatial_mask_prob={self.max_spatial_mask_prob}"
        )


def extract_masked_elements(seqs: Tensor, temporal_mask: Tensor) -> Tensor:
    """Extract masked elements from ``seqs``.

    :param seqs:
        The sequences. *Shape:* :math:`(N,S,M)`, where :math:`N` is the batch
        size, :math:`S` is the sequence length, and :math:`M` is the
        dimensionality of the model.
    :param temporal_mask:
        The temporal mask. *Shape:* :math:`(N,S)`, where :math:`N` is the batch
        size and :math`S` is the sequence length.
    """
    batch_size = seqs.size(0)

    # (N, S, M) -> (N x T, M)
    seqs = seqs[temporal_mask]

    # (N x T, M) -> (N, T, M)
    return seqs.unflatten(0, (batch_size, -1))  # type: ignore[no-any-return]
