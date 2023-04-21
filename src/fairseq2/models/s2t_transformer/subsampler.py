# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Final, Optional, Sequence, Tuple, final

import torch
from overrides import final as finaloverride
from torch import Tensor
from torch.nn import GLU, Conv1d, Module, Sequential


class FbankSubsampler(Module, ABC):
    """Subsamples log-mel filterbanks and embeds them in a latent space."""

    embed_dim: int

    def __init__(self, embed_dim: int) -> None:
        """
        :param embed_dim:
            The dimensionality of returned embeddings.
        """
        super().__init__()

        self.embed_dim = embed_dim

    @abstractmethod
    def forward(
        self, fbanks: Tensor, num_frames: Optional[Tensor]
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        :param fbanks:
            The log-mel filterbanks to subsample. *Shape:* :math:`(N,F,C)`, or
            :math:`(F,C)` when unbatched, where :math:`N` is the batch size,
            :math:`F` is the number of frames, and :math:`C` is the number of
            channels.
        :param num_frames:
            An array where each element represents the number of frames of the
            filterbank at the same index in ``fbanks``. *Shape:* :math:`(N)`,
            :math:`(N,1)`, or :math:`()` when unbatched, where :math:`N` is the
            batch size.

        :returns:
            - The audio embeddings, subsampled from ``fbanks``. *Shape:*
              :math:`(N,S,E)`, or :math:`(S,E)` when unbatched, where :math:`N`
              is the batch size, :math:`S` is the sequence length, and :math:`E`
              is the embedding size.
            - The sequence lengths corresponding to the returned audio
              embeddings. *Shape:* :math:`(N)`, or :math:`()` when unbatched,
              where :math:`N` is the batch size.
        """

    def extra_repr(self) -> str:
        """:meta private:"""
        return f"embed_dim={self.embed_dim}"


@final
class Conv1dFbankSubsampler(FbankSubsampler):
    """Represents a 1D convolutional subsampler as described in Section 2.1 of
    :cite:t:`https://doi.org/10.48550/arxiv.1911.08460`."""

    # All convolutions use the same stride length.
    stride: Final[int] = 2

    layers: Sequential

    def __init__(
        self,
        num_channels: int,
        inner_dim: int,
        embed_dim: int,
        kernel_sizes: Optional[Sequence[int]] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """
        :param num_channels:
            The number of channels of input log-mel filterbanks.
        :param inner_dim:
            The output dimensionality of the intermediate 1D convolutions.
        :param embed_dim:
            The dimensionality of returned embeddings.
        :param kernel_sizes:
            The kernel size of each 1D convolution.
        """
        super().__init__(embed_dim)

        if kernel_sizes is None:
            kernel_sizes = [3, 3]

        self.layers = Sequential()

        last_layer = len(kernel_sizes) - 1

        for i, kernel_size in enumerate(kernel_sizes):
            layer = Sequential()

            if i == 0:
                inp_dim = num_channels
            else:
                inp_dim = inner_dim // 2

            if i == last_layer:
                out_dim = embed_dim * 2
            else:
                out_dim = inner_dim

            conv = Conv1d(
                inp_dim,
                out_dim,
                kernel_size,
                stride=self.stride,
                padding=kernel_size // 2,
                device=device,
                dtype=dtype,
            )

            layer.add_module("conv", conv)
            layer.add_module("activation", GLU(dim=-2))

            self.layers.append(layer)

    @finaloverride
    def forward(
        self, fbanks: Tensor, num_frames: Optional[Tensor]
    ) -> Tuple[Tensor, Optional[Tensor]]:
        # Apply the convolution along the temporal dimension (i.e. along the
        # sequence).
        # (N, F, C) -> (N, C, F)
        x = fbanks.transpose(-1, -2)

        # (N, C, F) -> (N, E, S)
        x = self.layers(x)

        # (N, E, S) -> (N, S, E)
        x = x.transpose(-1, -2)

        if num_frames is None:
            return x, None
        else:
            # Since we contracted the temporal dimension, we should re-compute
            # the sequence lengths.
            return x, self._compute_seq_lens(num_frames)

    def _compute_seq_lens(self, num_frames: Tensor) -> Tensor:
        seq_lens = num_frames.clone()

        for _ in range(len(self.layers)):
            seq_lens = (((seq_lens - 1) / self.stride) + 1.0).floor()

        return seq_lens.type(num_frames.dtype)
