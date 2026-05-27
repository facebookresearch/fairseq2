# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import final

import torch
from torch import Tensor
from torch.nn import Module
from typing_extensions import override

from fairseq2.data_type import DataType
from fairseq2.device import Device
from fairseq2.models.gemma4.audio.config import Gemma4AudioConfig
from fairseq2.models.gemma4.audio.conformer import Gemma4ConformerEncoder
from fairseq2.models.gemma4.audio.subsample import Gemma4SubsampleConvProjection
from fairseq2.nn import BatchLayout


@final
class Gemma4AudioTower(Module):
    """Gemma4 audio tower for processing mel-spectrograms.

    Pipeline:
    1. Mel-spectrogram (N, T, 128) -> Subsample (4x downsample) -> (N, T/4, 1024)
    2. Conformer encoder (12 layers, NO reduction) -> (N, T/4, 1024)
    3. Output projection (Linear with bias) -> (N, T/4, 1536)

    The output_proj is the only layer in the audio tower with bias.
    Unlike Gemma3n, there is no reduction factor in the conformer encoder.
    """

    subsample: Gemma4SubsampleConvProjection
    encoder: Gemma4ConformerEncoder
    output_proj: torch.nn.Linear

    def __init__(
        self,
        audio_config: Gemma4AudioConfig,
        *,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        super().__init__()

        self.subsample = Gemma4SubsampleConvProjection(
            input_feat_size=audio_config.input_feat_size,
            hidden_size=audio_config.hidden_size,
            conv_channel_sizes=audio_config.subsampling_conv_channels,
            device=device,
            dtype=dtype,
        )

        self.encoder = Gemma4ConformerEncoder(
            audio_config,
            device=device,
            dtype=dtype,
        )

        # Output projection: hidden_size -> output_proj_dims (with bias)
        self.output_proj = torch.nn.Linear(
            audio_config.hidden_size,
            audio_config.output_proj_dims,
            bias=True,
            device=device,
            dtype=dtype,
        )

    @override
    def forward(self, features: Tensor) -> Tensor:
        """
        :param features: Mel-spectrogram. *Shape:* :math:`(N,T,F)` where F=128.
        :returns: Projected features. *Shape:* :math:`(N,T/4,D)` where D=output_proj_dims.
        """
        batch_size = features.size(0)

        # Subsample: (N, T, 128) -> (N, T/4, 1024)
        features = self.subsample(features)

        downsampled_len = features.size(1)
        layout = BatchLayout(
            (batch_size, downsampled_len),
            seq_lens=[downsampled_len] * batch_size,
        )

        # Conformer encode (NO reduction): (N, T/4, 1024) -> (N, T/4, 1024)
        features = self.encoder(features, layout)

        # Output projection: (N, T/4, 1024) -> (N, T/4, 1536)
        features = self.output_proj(features)

        return features
