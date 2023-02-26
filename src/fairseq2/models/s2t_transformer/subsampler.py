# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Optional, Sequence, Tuple, final

import torch
import torch.nn.functional as F
from overrides import final as finaloverride
from torch import Tensor
from torch.nn import Conv1d, Module, ModuleList


class FbankSubsampler(Module, ABC):
    """Subsamples log-mel filterbanks and embeds them in a latent space for use
    in sequence encoding and decoding."""

    embedding_dim: int

    def __init__(self, embedding_dim: int) -> None:
        """
        :param embedding_dim:
            The dimensionality of returned embeddings.
        """
        super().__init__()

        self.embedding_dim = embedding_dim

    @abstractmethod
    def forward(
        self, fbanks: Tensor, num_frames: Tensor
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        :param fbanks:
            The log-mel filterbanks to subsample. *Shape:* :math:`(N,F,C)`, or
            :math:`(F,C)` when unbatched, where :math:`N` is the batch size,
            :math:`F` is the number of frames, and :math:`C` is the number of
            channels.
        :param num_frames:
            An array where each entry defines the number of frames of the
            filterbank at the same index in ``fbanks``. *Shape:* :math:`(N)`,
            :math:`(N,1)`, or :math:`()` when unbatched, where :math:`N` is the
            batch size.

        :returns:
            - The audio embeddings to pass to the encoder or decoder. *Shape:*
              :math:`(N,S,E)`, or :math:`(S,E)` when unbatched, where :math:`N`
              is the batch size, :math:`S` is the sequence length, and :math:`E`
              is the embedding size.
            - The sequence lengths corresponding to the returned audio
              embeddings. *Shape:* :math:`(N)`, or :math:`()` when unbatched,
              where :math:`N` is the batch size.
        """

    def extra_repr(self) -> str:
        """:meta private:"""
        return f"embedding_dim={self.embedding_dim}"


@final
class Conv1dFbankSubsampler(FbankSubsampler):
    """Represents a 1D convolutional subsampler as described in Section 2.1 of
    :cite:t:`https://doi.org/10.48550/arxiv.1911.08460`."""

    convs: ModuleList

    def __init__(
        self,
        num_channels: int,
        inner_dim: int,
        embedding_dim: int,
        kernel_sizes: Optional[Sequence[int]] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """
        :param num_channels:
            The number of channels of input log-mel filterbanks.
        :param inner_dim:
            The dimensionality of the intermediate 1D convolution layers.
        :param embedding_dim:
            The dimensionality of returned embeddings.
        :param kernel_sizes:
            The kernel size of each 1D convolutional layer.
        """
        super().__init__(embedding_dim)

        if kernel_sizes is None:
            kernel_sizes = [3, 3]

        if not kernel_sizes:
            raise ValueError("`kernel_sizes` must contain at least one element.")

        last_layer = len(kernel_sizes) - 1

        convs = [
            Conv1d(
                num_channels if i == 0 else inner_dim // 2,
                inner_dim if i < last_layer else embedding_dim * 2,
                kernel_size,
                stride=2,
                padding=kernel_size // 2,
                device=device,
                dtype=dtype,
            )
            for i, kernel_size in enumerate(kernel_sizes)
        ]

        self.convs = ModuleList(convs)

    @finaloverride
    def forward(
        self, fbanks: Tensor, num_frames: Tensor
    ) -> Tuple[Tensor, Optional[Tensor]]:
        # Apply the convolution along the temporal dimension (i.e. along the
        # sequence).
        # (N, F, C) -> (N, C, F)
        x = fbanks.transpose(1, 2)

        for conv in self.convs:
            x = conv(x)

            x = F.glu(x, dim=1)

        # (N, E, F), -> (N, F, E)
        x = x.transpose(1, 2)

        # Since we contracted the temporal dimension, we should re-compute the
        # sequence lengths.
        seq_lens = self._compute_seq_lens(num_frames)

        return x, seq_lens

    def _compute_seq_lens(self, num_frames: Tensor) -> Tensor:
        seq_lens = num_frames.clone()

        for _ in range(len(self.convs)):
            seq_lens = (((seq_lens - 1) / 2.0) + 1.0).floor().type(seq_lens.dtype)

        return seq_lens
