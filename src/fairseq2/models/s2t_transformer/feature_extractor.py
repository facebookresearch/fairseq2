# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Final, Optional, Sequence, Tuple, final

from torch import Tensor
from torch.nn import GLU, Conv1d, Sequential

from fairseq2.models.feature_extractor import SequenceFeatureExtractor
from fairseq2.nn.padding import PaddingMask
from fairseq2.typing import DataType, Device, finaloverride


@final
class Conv1dFbankSubsampler(SequenceFeatureExtractor):
    """Extracts features from log-mel filterbanks and embeds them in a latent
    space as described in Section 2.1 of
    :cite:t:`https://doi.org/10.48550/arxiv.1911.08460`.
    """

    # All convolutions use the same stride.
    stride: Final[int] = 2

    layers: Sequential

    def __init__(
        self,
        num_channels: int,
        inner_dim: int,
        feature_dim: int,
        *,
        kernel_sizes: Optional[Sequence[int]] = None,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
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
        super().__init__(feature_dim)

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
                stride=self.stride,
                padding=kernel_size // 2,
                device=device,
                dtype=dtype,
            )

            layer.add_module("conv", conv)
            layer.add_module("activation", GLU(dim=1))

            self.layers.append(layer)

    @finaloverride
    def forward(
        self, seqs: Tensor, padding_mask: Optional[PaddingMask]
    ) -> Tuple[Tensor, Optional[PaddingMask]]:
        """See the base :meth:`SequenceFeatureExtractor.forward`.

        :param seqs:
            The input log-mel filterbanks. *Shape:* :math:`(N,S,C)`, where
            :math:`N` is the batch size, :math:`S` is the number of frames, and
            :math:`C` is the number of channels.
        """
        # Apply the convolution along the temporal dimension (i.e. along the
        # sequence).
        # (N, S, C) -> (N, C, S)
        seqs = seqs.transpose(1, 2)

        # (N, C, S) -> (N, F, S_out)
        features = self.layers(seqs)

        # (N, F, S_out) -> (N, S_out, F)
        features = features.transpose(1, 2)

        # Since we contracted the temporal dimension, we should re-compute
        # the sequence lengths.
        if padding_mask is not None:
            seq_lens = self._contract_seq_lens(padding_mask.seq_lens)

            padding_mask = PaddingMask(seq_lens, batch_seq_len=features.size(1))

        return features, padding_mask

    def _contract_seq_lens(self, num_frames: Tensor) -> Tensor:
        seq_lens = num_frames.clone()

        for _ in range(len(self.layers)):
            seq_lens = (((seq_lens - 1) / self.stride) + 1.0).floor()

        return seq_lens.type_as(num_frames)
