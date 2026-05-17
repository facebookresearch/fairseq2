# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import final

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Conv2d, LayerNorm, Module, ReLU
from typing_extensions import override

from fairseq2.data_type import DataType
from fairseq2.device import Device
from fairseq2.nn.projection import Linear


@final
class Gemma4SubsampleConvProjection(Module):
    """Subsample mel-spectrogram and project to audio encoder hidden size.

    Applies two 2D convolution blocks with LayerNorm (not CumulativeGroupNorm
    as in Gemma3n) to downsample the mel-spectrogram by 4x in both time and
    frequency dimensions, then projects to the audio encoder hidden size.

    Uses symmetric padding (padding=1 on freq), matching HF's Gemma4 reference.
    """

    conv_0: Conv2d
    norm_0: LayerNorm
    conv_1: Conv2d
    norm_1: LayerNorm
    proj: Linear
    activation: ReLU

    def __init__(
        self,
        input_feat_size: int = 128,
        hidden_size: int = 1024,
        conv_channel_sizes: tuple[int, int] = (128, 32),
        *,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        super().__init__()

        ch0, ch1 = conv_channel_sizes

        # Gemma4 uses kernel_size=3, stride=2, padding=1 (symmetric)
        self.conv_0 = Conv2d(
            in_channels=1,
            out_channels=ch0,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
            device=device,
            dtype=dtype,
        )

        # Output frequency dim after conv_0: (input_feat_size + 2*1 - 3) // 2 + 1
        f_out_0 = (input_feat_size + 2 - 3) // 2 + 1  # = input_feat_size // 2

        # LayerNorm(ch0, elementwise_affine=True, bias=False, eps=1e-6)
        # HF uses eps=config.rms_norm_eps (1e-6), NOT the PyTorch default 1e-5.
        self.norm_0 = LayerNorm(
            ch0,
            eps=1e-6,
            elementwise_affine=True,
            bias=False,
            device=device,
            dtype=dtype,
        )

        self.conv_1 = Conv2d(
            in_channels=ch0,
            out_channels=ch1,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
            device=device,
            dtype=dtype,
        )

        # Output frequency dim after conv_1
        f_out_1 = (f_out_0 + 2 - 3) // 2 + 1  # = f_out_0 // 2

        self.norm_1 = LayerNorm(
            ch1,
            eps=1e-6,
            elementwise_affine=True,
            bias=False,
            device=device,
            dtype=dtype,
        )

        # Flattened feature dim: freq_dim * channels
        flattened_size = f_out_1 * ch1

        self.proj = Linear(
            flattened_size,
            hidden_size,
            bias=False,
            device=device,
            dtype=dtype,
        )

        self.activation = ReLU()

    @override
    def forward(self, features: Tensor) -> Tensor:
        """
        :param features: Mel-spectrogram ``[B, T, F]`` where F=128.
        :returns: Subsampled features ``[B, T/4, H]`` where H=hidden_size.
        """
        batch_size = features.size(0)

        # [B, T, F] -> [B, 1, T, F]
        x = features.unsqueeze(1).to(self.conv_0.weight.dtype)

        # Block 0: Conv2d + LayerNorm + ReLU
        x = self.conv_0(x)  # [B, C0, T/2, F/2]
        x = x.permute(0, 2, 3, 1)  # [B, T/2, F/2, C0]
        x = self.norm_0(x)
        x = self.activation(x)
        x = x.permute(0, 3, 1, 2)  # [B, C0, T/2, F/2]

        # Block 1: Conv2d + LayerNorm + ReLU
        x = self.conv_1(x)  # [B, C1, T/4, F/4]
        x = x.permute(0, 2, 3, 1)  # [B, T/4, F/4, C1]
        x = self.norm_1(x)
        x = self.activation(x)

        # Flatten freq and channel: [B, T/4, F/4 * C1]
        x = x.reshape(batch_size, x.size(1), -1)

        # Project to hidden_size
        x = self.proj(x)

        return x
