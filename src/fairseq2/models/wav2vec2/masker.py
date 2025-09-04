# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, final

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module, Parameter
from typing_extensions import override

from fairseq2.data_type import DataType
from fairseq2.device import Device
from fairseq2.error import InternalError
from fairseq2.nn import BatchLayout
from fairseq2.nn.utils.mask import RowMaskFactory, compute_row_mask
from fairseq2.nn.utils.module import get_name_or_self


class Wav2Vec2Masker(Module, ABC):
    """Masks extracted wav2vec 2.0 features."""

    @abstractmethod
    def forward(self, seqs: Tensor, seqs_layout: BatchLayout) -> tuple[Tensor, Tensor]:
        """
        :param seqs:
            The sequences to mask. *Shape:* :math:`(N,S,M)`, where :math:`N` is
            the batch size, :math:`S` is the sequence length, and :math:`M` is
            the dimensionality of the model.

        :returns:
            - The input sequences with mask applied. *Shape:* Same as ``seqs``.
            - The temporal mask that has been applied to ``seqs``. *Shape:*
              :math:`(N,S)`, where :math:`N` is the batch size and :math`S` is
              the sequence length.
        """

    if TYPE_CHECKING:
        __call__ = forward

    @staticmethod
    def extract_masked_elements(seqs: Tensor, temporal_mask: Tensor) -> Tensor:
        """
        Extracts masked elements from ``seqs``.

        :param seqs: The sequences. *Shape:* :math:`(N,S,M)`, where :math:`N` is
            the batch size, :math:`S` is the sequence length, and :math:`M` is
            the dimensionality of the model.
        :param temporal_mask: The temporal mask. *Shape:* :math:`(N,S)`, where
            :math:`N` is the batch size and :math`S` is the sequence length.
        """
        batch_size = seqs.size(0)

        # (N, S, M) -> (N x T, M)
        seqs = seqs[temporal_mask]

        # (N x T, M) -> (N, T, M)
        return seqs.unflatten(0, (batch_size, -1))  # type: ignore[no-any-return]


@final
class StandardWav2Vec2Masker(Wav2Vec2Masker):
    """Masks extracted wav2vec 2.0 features as described in Section 3.1 of
    :cite:t:`https://doi.org/10.48550/arxiv.2006.11477`."""

    def __init__(
        self,
        model_dim: int,
        temporal_span_len: int = 10,
        max_temporal_mask_prob: float = 0.65,
        min_num_temporal_mask_spans: int = 2,
        spatial_span_len: int = 10,
        max_spatial_mask_prob: float = 0.0,
        min_num_spatial_mask_spans: int = 2,
        *,
        mask_factory: RowMaskFactory | None = None,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        """
        :param model_dim:
            The dimensionality of the model.
        :param temporal_span_len:
            The length of each temporal mask span that is applied over time
            steps.
        :param max_temporal_mask_prob:
            The maximum probability of masking a time step. Note that, due to
            mask span overlap, the effective probability will be lower.
        :param spatial_span_len:
            The length of each spatial mask span that is applied over features.
        :param max_spatial_mask_prob:
            The maximum probability of masking a feature. Note that, due to mask
            span overlap, the effective probability will be lower.
        :param mask_factory:
            The row mask factory. If ``None``, :func:`compute_row_mask` will be
            used.
        """
        super().__init__()

        if max_temporal_mask_prob <= 0.0:
            raise ValueError("`max_temporal_mask_prob` must be greater than 0.")

        self.temporal_mask_embed = Parameter(
            torch.empty((model_dim,), device=device, dtype=dtype)
        )

        self.temporal_span_len = temporal_span_len
        self.max_temporal_mask_prob = max_temporal_mask_prob
        self.min_num_temporal_mask_spans = min_num_temporal_mask_spans

        self.spatial_span_len = spatial_span_len
        self.max_spatial_mask_prob = max_spatial_mask_prob
        self.min_num_spatial_mask_spans = min_num_spatial_mask_spans

        self.mask_factory = mask_factory or compute_row_mask

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.temporal_mask_embed)

    @override
    def forward(self, seqs: Tensor, seqs_layout: BatchLayout) -> tuple[Tensor, Tensor]:
        if seqs_layout.packed:
            raise ValueError("`seqs` must not be a packed batch.")

        batch_size, seq_len, model_dim = seqs.shape

        # Temporal mask over time steps.
        temporal_mask = self.mask_factory(
            shape=(batch_size, seq_len),
            span_len=self.temporal_span_len,
            max_mask_prob=self.max_temporal_mask_prob,
            row_lens=seqs_layout.seq_lens_pt,
            min_num_spans=self.min_num_temporal_mask_spans,
            device=seqs.device,
        )

        if temporal_mask is None:
            raise InternalError("`temporal_mask` is `None`.")

        seqs[temporal_mask] = self.temporal_mask_embed.type_as(seqs)

        if self.max_spatial_mask_prob > 0.0:
            # Spatial mask over features.
            # (N, M)
            spatial_mask = self.mask_factory(
                shape=(batch_size, model_dim),
                span_len=self.spatial_span_len,
                max_mask_prob=self.max_spatial_mask_prob,
                min_num_spans=self.min_num_spatial_mask_spans,
                device=seqs.device,
            )

            if spatial_mask is None:
                raise InternalError("`spatial_mask` is `None`.")

            # (N, M) -> (N, S, M)
            spatial_mask = spatial_mask.unsqueeze(1).expand(-1, seq_len, -1)

            seqs[spatial_mask] = 0.0

        return seqs, temporal_mask

    @override
    def extra_repr(self) -> str:
        """:meta private:"""
        s = (
            f"temporal_span_len={self.temporal_span_len}, "
            f"max_temporal_mask_prob={self.max_temporal_mask_prob}, "
            f"min_num_temporal_mask_spans={self.min_num_temporal_mask_spans}, "
            f"spatial_span_len={self.spatial_span_len}, "
            f"max_spatial_mask_prob={self.max_spatial_mask_prob}, "
            f"min_num_spatial_mask_spans={self.min_num_spatial_mask_spans}"
        )

        if self.mask_factory is not compute_row_mask:
            mask_factory = get_name_or_self(self.mask_factory)

            s = f"{s}, mask_factory={mask_factory}"

        return s
