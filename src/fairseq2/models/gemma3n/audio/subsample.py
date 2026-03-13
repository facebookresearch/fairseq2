# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import final

from torch import Tensor
from torch.nn import Conv2d, GroupNorm, Module, SiLU
from typing_extensions import override

from fairseq2.data_type import DataType
from fairseq2.device import Device
from fairseq2.nn.projection import Linear


@final
class Gemma3nSubsampleConvProjection(Module):
    """Subsample mel-spectrogram and project to audio encoder hidden size.

    Applies two 2D convolution blocks with group normalization to downsample
    the mel-spectrogram by 4x in both time and frequency dimensions, then
    projects to the audio encoder hidden size via linear layer.
    """

    conv_0: Conv2d
    norm_0: GroupNorm
    conv_1: Conv2d
    norm_1: GroupNorm
    proj: Linear
    activation: SiLU

    def __init__(
        self,
        input_feat_size: int,
        hidden_size: int,
        conv_channel_sizes: tuple[int, int],
        conv_kernel_sizes: tuple[tuple[int, int], tuple[int, int]],
        conv_strides: tuple[tuple[int, int], tuple[int, int]],
        group_norm_eps: float = 1e-3,
        *,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        """
        :param input_feat_size: Number of mel-spectrogram frequency bins (128).
        :param hidden_size: Audio encoder hidden dimension (1536).
        :param conv_channel_sizes: Output channels for each conv layer (128, 32).
        :param conv_kernel_sizes: Kernel sizes (time, freq) for each conv layer.
        :param conv_strides: Stride sizes (time, freq) for each conv layer.
        :param group_norm_eps: Epsilon for group normalization.
        """
        super().__init__()

        ch0, ch1 = conv_channel_sizes
        k0, k1 = conv_kernel_sizes
        s0, s1 = conv_strides

        self.conv_0 = Conv2d(
            in_channels=1,
            out_channels=ch0,
            kernel_size=k0,
            stride=s0,
            padding=(k0[0] // 2, k0[1] // 2),
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.norm_0 = GroupNorm(
            num_groups=1,
            num_channels=ch0,
            eps=group_norm_eps,
            device=device,
            dtype=dtype,
        )
        # HF's CumulativeGroupNorm has weight but no bias
        self.norm_0.register_parameter("bias", None)

        self.conv_1 = Conv2d(
            in_channels=ch0,
            out_channels=ch1,
            kernel_size=k1,
            stride=s1,
            padding=(k1[0] // 2, k1[1] // 2),
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.norm_1 = GroupNorm(
            num_groups=1,
            num_channels=ch1,
            eps=group_norm_eps,
            device=device,
            dtype=dtype,
        )
        # HF's CumulativeGroupNorm has weight but no bias
        self.norm_1.register_parameter("bias", None)

        downsampled_freq = input_feat_size // (s0[1] * s1[1])
        flattened_size = ch1 * downsampled_freq

        self.proj = Linear(
            flattened_size,
            hidden_size,
            bias=False,
            device=device,
            dtype=dtype,
        )

        self.activation = SiLU()

    @override
    def forward(self, features: Tensor) -> Tensor:
        """
        :param features: Mel-spectrogram. *Shape:* :math:`(N,T,F)` where F=128.
        :returns: Subsampled features. *Shape:* :math:`(N,T/4,H)` where H=1536.
        """
        batch_size = features.size(0)

        x = features.unsqueeze(1)

        x = self.conv_0(x)
        x = self.norm_0(x)
        x = self.activation(x)

        x = self.conv_1(x)
        x = self.norm_1(x)
        x = self.activation(x)

        x = x.permute(0, 2, 3, 1)
        x = x.reshape(batch_size, x.size(1), -1)

        x = self.proj(x)

        return x
