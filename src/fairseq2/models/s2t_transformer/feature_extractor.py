# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Final, final

from torch import Tensor
from torch.nn import GLU, Conv1d, Sequential
from typing_extensions import override

from fairseq2.data_type import DataType
from fairseq2.device import Device
from fairseq2.models.feature_extractor import SequenceFeatureExtractor
from fairseq2.nn import BatchLayout


@final
class Conv1dFbankSubsampler(SequenceFeatureExtractor):
    """Extracts features from log-mel filterbanks and embeds them in a latent
    space as described in Section 2.1 of
    :cite:t:`https://doi.org/10.48550/arxiv.1911.08460`.
    """

    STRIDE: Final[int] = 2  # All convolutions use the same stride.

    def __init__(
        self,
        num_channels: int,
        inner_dim: int,
        feature_dim: int,
        *,
        kernel_sizes: Sequence[int] | None = None,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        """
        :param num_channels:
            The number of channels of input log-mel filterbanks.
        :param inner_dim:
            The output dimensionality of the intermediate 1D convolutions.
        :param feature_dim:
            The dimensionality of extracted features.
        :param kernel_sizes:
            The kernel size of each 1D convolution.
        """
        super().__init__()

        if kernel_sizes is None:
            kernel_sizes = [3, 3]

        self.layers = Sequential()

        last_layer = len(kernel_sizes) - 1

        for i, kernel_size in enumerate(kernel_sizes):
            layer = Sequential()

            if i == 0:
                layer_input_dim = num_channels
            else:
                layer_input_dim = inner_dim // 2

            if i == last_layer:
                layer_output_dim = feature_dim * 2
            else:
                layer_output_dim = inner_dim

            conv = Conv1d(
                layer_input_dim,
                layer_output_dim,
                kernel_size,
                stride=self.STRIDE,
                padding=kernel_size // 2,
                device=device,
                dtype=dtype,
            )

            layer.add_module("conv", conv)

            layer.add_module("activation", GLU(dim=1))

            self.layers.append(layer)

    @override
    def forward(
        self, seqs: Tensor, seqs_layout: BatchLayout
    ) -> tuple[Tensor, BatchLayout]:
        """See the base :meth:`SequenceFeatureExtractor.forward`.

        :param seqs:
            The input log-mel filterbanks. *Shape:* :math:`(N,S,C)`, where
            :math:`N` is the batch size, :math:`S` is the number of frames, and
            :math:`C` is the number of channels.
        """
        if seqs_layout.padded:
            raise ValueError("`seqs` must not be a packed batch.")

        # Apply the convolution along the temporal dimension (i.e. along the
        # sequence).
        # (N, S, C) -> (N, C, S)
        seqs = seqs.transpose(1, 2)

        # (N, C, S) -> (N, F, S_out)
        seqs = self.layers(seqs)

        # (N, F, S_out) -> (N, S_out, F)
        seqs = seqs.transpose(1, 2)

        if seqs_layout.padded:
            # Since we contracted the temporal dimension, we should re-compute
            # the sequence lengths.
            seq_lens = self._contract_seq_lens(seqs_layout.seq_lens)
        else:
            seq_lens = None

        seqs_layout = BatchLayout.of(seqs, seq_lens)

        return seqs, seqs_layout

    def _contract_seq_lens(self, seq_lens: Sequence[int]) -> list[int]:
        seq_lens = list(seq_lens)

        for _ in range(len(self.layers)):
            for i in range(len(seq_lens)):
                seq_lens[i] = math.floor(((seq_lens[i] - 1) / self.STRIDE) + 1.0)

        return seq_lens
