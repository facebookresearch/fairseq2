from __future__ import annotations

from typing import final

import torch
from torch import Tensor
from torch.nn import Module

from fairseq2.nn.utils.mask import compute_row_mask
from fairseq2.nn.padding import PaddingMask
from fairseq2.typing import DataType, Device


@final
class RandomNoiseMasker(Module):
    """Masks extracted features as described in Section 3.1 of
    :cite:t:`https://doi.org/10.48550/arxiv.2006.11477`."""

    temporal_span_len: int
    max_temporal_mask_prob: float
    spatial_span_len: int
    noise_mean: float
    noise_standard_deviation: float
    max_spatial_mask_prob: float

    def __init__(
        self,
        model_dim: int,
        temporal_span_len: int = 10,
        max_temporal_mask_prob: float = 0.65,
        min_num_temporal_mask_spans: int = 2,
        spatial_span_len: int = 10,
        max_spatial_mask_prob: float = 0.0,
        min_num_spatial_mask_spans: int = 2,
        mask_overlap_strategy: str = "no",  # remove_masks (fs2 default), add_masks_jc (additive, v1), add_masks_mike (additive, v2), roll (Alex's suggestion)
        noise_mean: float = 0.0,
        noise_standard_deviation: float = 0.1,
        *,
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
        """
        super().__init__()

        if max_temporal_mask_prob == 0.0:
            raise ValueError("`max_temporal_mask_prob` must be greater than 0.")

        self.temporal_span_len = temporal_span_len
        self.max_temporal_mask_prob = max_temporal_mask_prob
        self.min_num_temporal_mask_spans = min_num_temporal_mask_spans

        self.noise_mean = torch.tensor(noise_mean)
        self.noise_standard_deviation = torch.tensor(noise_standard_deviation)

        self.spatial_span_len = spatial_span_len
        self.max_spatial_mask_prob = max_spatial_mask_prob
        self.min_num_spatial_mask_spans = min_num_spatial_mask_spans
        self.mask_overlap_strategy = mask_overlap_strategy

    def forward(
        self, seqs: Tensor, padding_mask: PaddingMask | None
    ) -> tuple[Tensor, Tensor]:
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
            min_num_spans=self.min_num_temporal_mask_spans,
            device=seqs.device,
        )

        assert temporal_mask is not None

        seqs[temporal_mask] = (
            self.noise_mean
            + torch.randn(
                seqs[temporal_mask].shape,
                device=seqs.device,
                dtype=seqs.dtype,
            )
            * self.noise_standard_deviation
        )

        if self.max_spatial_mask_prob > 0.0:
            # Spatial mask over features.
            # (N, M)
            spatial_mask = compute_row_mask(
                shape=(batch_size, model_dim),
                span_len=self.spatial_span_len,
                max_mask_prob=self.max_spatial_mask_prob,
                min_num_spans=self.min_num_spatial_mask_spans,
                mask_overlap_strategy=self.mask_overlap_strategy,
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
            f"min_num_temporal_mask_spans={self.min_num_temporal_mask_spans}, "
            f"spatial_span_len={self.spatial_span_len}, "
            f"max_spatial_mask_prob={self.max_spatial_mask_prob}, "
            f"min_num_spatial_mask_spans={self.min_num_spatial_mask_spans}"
        )
