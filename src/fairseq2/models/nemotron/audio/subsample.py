# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Conv2D subsampling for the Parakeet audio encoder.

Performs 8x temporal downsampling of mel spectrograms via three stages:
  Stage 1: Conv2d(1 → C, 3×3, stride 2, pad 1) → ReLU
  Stage 2: DepthwiseConv2d(C, 3×3, stride 2, pad 1) → PointwiseConv2d(C → C, 1×1) → ReLU
  Stage 3: DepthwiseConv2d(C, 3×3, stride 2, pad 1) → PointwiseConv2d(C → C, 1×1) → ReLU

Followed by a linear projection from C × (mel_bins / 8) to hidden_size.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, final

import torch
from torch import Tensor
from torch.nn import Conv2d, Linear, Module, ReLU, Sequential
from typing_extensions import override

from fairseq2.data_type import DataType
from fairseq2.device import Device


@final
class ParakeetSubsamplingConv2D(Module):
    """Conv2D subsampling for Parakeet/FastConformer audio encoder.

    Converts mel spectrograms [B, T, num_mel_bins] into subsampled features
    [B, T/8, hidden_size] through three stages of stride-2 convolutions.

    The three stages follow NVIDIA NeMo's Conv2dSubsampling pattern:
    - Stage 1: standard Conv2d
    - Stages 2-3: depthwise-separable Conv2d
    """

    def __init__(
        self,
        num_mel_bins: int = 128,
        hidden_size: int = 1024,
        conv_channels: int = 256,
        *,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        super().__init__()

        self.num_mel_bins = num_mel_bins

        # Stage 1: regular Conv2d
        # Input: [B, 1, T, mel_bins], Output: [B, C, T/2, mel_bins/2]
        stage1 = Conv2d(
            1,
            conv_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=True,
            device=device,
            dtype=dtype,
        )

        # Stage 2: depthwise + pointwise Conv2d
        # Input: [B, C, T/2, mel_bins/2], Output: [B, C, T/4, mel_bins/4]
        dw2 = Conv2d(
            conv_channels,
            conv_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=conv_channels,
            bias=True,
            device=device,
            dtype=dtype,
        )
        pw2 = Conv2d(
            conv_channels,
            conv_channels,
            kernel_size=1,
            bias=True,
            device=device,
            dtype=dtype,
        )

        # Stage 3: depthwise + pointwise Conv2d
        # Input: [B, C, T/4, mel_bins/4], Output: [B, C, T/8, mel_bins/8]
        dw3 = Conv2d(
            conv_channels,
            conv_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=conv_channels,
            bias=True,
            device=device,
            dtype=dtype,
        )
        pw3 = Conv2d(
            conv_channels,
            conv_channels,
            kernel_size=1,
            bias=True,
            device=device,
            dtype=dtype,
        )

        self.layers = Sequential(
            stage1,      # 0
            ReLU(),      # 1
            dw2,         # 2
            pw2,         # 3
            ReLU(),      # 4
            dw3,         # 5
            pw3,         # 6
            ReLU(),      # 7
        )

        # After 3× stride-2 convolutions, frequency dim = mel_bins // 8
        freq_dim = num_mel_bins // 8  # 128 // 8 = 16
        self.linear = Linear(
            conv_channels * freq_dim,
            hidden_size,
            bias=True,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Subsample mel spectrogram features.

        :param x:
            Mel spectrogram features. *Shape:* ``[B, T, num_mel_bins]``.

        :returns:
            Subsampled features. *Shape:* ``[B, T/8, hidden_size]``.
        """
        # [B, T, mel_bins] -> [B, 1, T, mel_bins]
        x = x.unsqueeze(1)

        # Apply 3-stage Conv2d subsampling
        # [B, 1, T, mel_bins] -> [B, C, T/8, mel_bins/8]
        x = self.layers(x)

        b, c, t, f = x.shape

        # [B, C, T/8, mel_bins/8] -> [B, T/8, C * mel_bins/8]
        x = x.permute(0, 2, 1, 3).reshape(b, t, c * f)

        # [B, T/8, C * mel_bins/8] -> [B, T/8, hidden_size]
        x = self.linear(x)

        return x

    if TYPE_CHECKING:
        __call__ = forward

    @override
    def extra_repr(self) -> str:
        """:meta private:"""
        return f"num_mel_bins={self.num_mel_bins}"
