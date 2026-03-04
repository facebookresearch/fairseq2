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
from fairseq2.models.gemma3n.audio import Gemma3nSubsampleConvProjection
from fairseq2.models.gemma3n.config import Gemma3nAudioConfig, Gemma3nConfig
from fairseq2.models.gemma3n.conformer import Gemma3nConformerEncoder
from fairseq2.models.gemma3n.multimodal_embedder import Gemma3nMultimodalEmbedder
from fairseq2.nn import BatchLayout


@final
class Gemma3nAudioTower(Module):
    """Gemma3n audio tower for processing mel-spectrograms to text space.

    Pipeline:
    1. Mel-spectrogram (N, T, 128) → Subsample (4x downsample) → (N, T/4, 1536)
    2. Conformer encoder (12 layers + 4x reduction) → (N, T/16, 1536)
    3. Multimodal embedder → Text space (N, T/16, 2048)
    """

    subsample: Gemma3nSubsampleConvProjection
    encoder: Gemma3nConformerEncoder
    embedder: Gemma3nMultimodalEmbedder

    def __init__(
        self,
        audio_config: Gemma3nAudioConfig,
        text_config: Gemma3nConfig,
        *,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        """
        :param audio_config: Audio tower configuration.
        :param text_config: Text model configuration for projection target.
        """
        super().__init__()

        self.subsample = Gemma3nSubsampleConvProjection(
            input_feat_size=audio_config.input_feat_size,
            hidden_size=audio_config.hidden_size,
            conv_channel_sizes=audio_config.sscp_conv_channel_size,
            conv_kernel_sizes=audio_config.sscp_conv_kernel_size,
            conv_strides=audio_config.sscp_conv_stride_size,
            group_norm_eps=audio_config.sscp_conv_group_norm_eps,
            device=device,
            dtype=dtype,
        )

        self.encoder = Gemma3nConformerEncoder(
            audio_config,
            device=device,
            dtype=dtype,
        )

        self.embedder = Gemma3nMultimodalEmbedder(
            audio_config,
            text_config,
            device=device,
            dtype=dtype,
        )

    @override
    def forward(
        self, features: Tensor, mask: Tensor | None = None
    ) -> tuple[Tensor, BatchLayout]:
        """
        :param features: Mel-spectrogram. *Shape:* :math:`(N,T,F)` where F=128.
        :param mask: Optional mask where True=masked (invalid). *Shape:* :math:`(N,T)`.
        :returns: Tuple of (projected features, layout).
                 - features: *Shape:* :math:`(N,T/16,H_{text})` where H_text=2048.
                 - layout: BatchLayout for the downsampled sequence.
        """
        batch_size = features.size(0)
        seq_len = features.size(1)

        # Subsample: (N, T, 128) → (N, T/4, 1536)
        features = self.subsample(features)

        # Subsample mask to match conformer input
        subsampled_mask = None
        if mask is not None:
            # Calculate subsample stride product
            time_stride = 1
            for stride_pair in [(2, 2), (2, 2)]:  # sscp_conv_stride_size default
                time_stride *= stride_pair[0]

            # Gather mask at subsampled indices
            t_sub = features.size(1)
            indices = torch.arange(t_sub, device=mask.device) * time_stride
            indices = torch.clamp(indices, max=mask.size(1) - 1)
            if indices.dim() == 1 and batch_size > 0:
                indices = indices.unsqueeze(0).expand(batch_size, -1)
            subsampled_mask = torch.gather(mask, 1, indices)

        # Create layout for downsampled sequence
        downsampled_len = features.size(1)
        layout = BatchLayout(
            (batch_size, downsampled_len),
            seq_lens=[downsampled_len] * batch_size,
        )

        # Conformer encode + reduction: (N, T/4, 1536) → (N, T/16, 1536)
        features = self.encoder(features, layout, subsampled_mask)

        # Update layout for reduced sequence
        reduced_len = features.size(1)
        layout = BatchLayout(
            (batch_size, reduced_len),
            seq_lens=[reduced_len] * batch_size,
        )

        # Project to text space: (N, T/16, 1536) → (N, T/16, 2048)
        features = self.embedder(features, is_soft=True)

        return features, layout
