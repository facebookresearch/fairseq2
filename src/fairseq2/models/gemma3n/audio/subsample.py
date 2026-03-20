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
from torch.nn import Conv2d, Module, Parameter, ReLU
from typing_extensions import override

from fairseq2.data_type import DataType
from fairseq2.device import Device
from fairseq2.nn.projection import Linear


@final
class CumulativeGroupNorm(Module):
    """Group normalization with cumulative statistics along the time dim.

    Computes running mean/variance cumulatively over dim=1 (time),
    reducing over all feature dimensions and channels. Matches HF's
    ``Gemma3nAudioCumulativeGroupNorm`` with ``num_groups=1``.

    Input shape: ``[B, T, *feature_dims, C]``.
    Only has a ``weight`` parameter (shape ``[C]``), no bias.
    """

    weight: Parameter
    num_channels: int
    feature_dims: tuple[int, ...]
    eps: float
    reduction_axes: tuple[int, ...]

    def __init__(
        self,
        num_channels: int,
        feature_dims: tuple[int, ...],
        eps: float = 1e-3,
        *,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.feature_dims = feature_dims
        self.eps = eps
        self.reduction_axes = tuple(range(2, 2 + len(feature_dims) + 1))
        self.weight = Parameter(torch.ones(num_channels, device=device, dtype=dtype))

    @override
    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: ``[B, T, *feature_dims, C]``
        :returns: Normalized tensor, same shape.
        """
        input_dtype = x.dtype
        x_fp32 = x.to(torch.float32)

        # Cumulative mean along time (dim=1).
        sum_at_t = x_fp32.sum(dim=self.reduction_axes, keepdim=True)
        cum_sum = torch.cumsum(sum_at_t, dim=1)

        count_at_t = torch.ones_like(x_fp32).sum(dim=self.reduction_axes, keepdim=True)
        cum_count = torch.cumsum(count_at_t, dim=1).clamp(min=1.0)

        cum_mean = cum_sum / cum_count

        # Cumulative variance along time (dim=1).
        sq_diff = (x_fp32 - cum_mean).pow(2)
        sq_diff_sum_at_t = sq_diff.sum(dim=self.reduction_axes, keepdim=True)
        cum_sq_diff = torch.cumsum(sq_diff_sum_at_t, dim=1)
        cum_var = cum_sq_diff / cum_count

        normalized = (x_fp32 - cum_mean) * torch.rsqrt(cum_var + self.eps)

        # Per-channel scale: [C] -> [1, ..., 1, C].
        scale_shape = [1] * (x.dim() - 1) + [self.num_channels]
        scale = self.weight.to(torch.float32).view(scale_shape)
        normalized = normalized * scale

        return normalized.to(input_dtype)


@final
class Gemma3nSubsampleConvProjection(Module):
    """Subsample mel-spectrogram and project to audio encoder hidden size.

    Applies two 2D convolution blocks with cumulative group normalization
    to downsample the mel-spectrogram by 4x in both time and frequency
    dimensions, then projects to the audio encoder hidden size.

    Uses reverse-causal padding (0, kernel_h-1) on time and symmetric
    (1, 1) on frequency, matching the HF reference implementation.
    """

    conv_0: Conv2d
    norm_0: CumulativeGroupNorm
    conv_1: Conv2d
    norm_1: CumulativeGroupNorm
    proj: Linear
    activation: ReLU

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
        super().__init__()

        ch0, ch1 = conv_channel_sizes
        k0, k1 = conv_kernel_sizes
        s0, s1 = conv_strides

        # Reverse-causal padding: (pad_F_left, pad_F_right, 0, kH-1)
        self._pad_0 = (1, 1, 0, k0[0] - 1)
        self._pad_1 = (1, 1, 0, k1[0] - 1)

        self.conv_0 = Conv2d(
            in_channels=1,
            out_channels=ch0,
            kernel_size=k0,
            stride=s0,
            padding=(0, 0),
            bias=False,
            device=device,
            dtype=dtype,
        )

        # f_out after conv_0: (input_feat + pad_left + pad_right - kW) // sW + 1
        f_out_0 = (input_feat_size + 2 - k0[1]) // s0[1] + 1

        self.norm_0 = CumulativeGroupNorm(
            num_channels=ch0,
            feature_dims=(f_out_0,),
            eps=group_norm_eps,
            device=device,
            dtype=dtype,
        )

        self.conv_1 = Conv2d(
            in_channels=ch0,
            out_channels=ch1,
            kernel_size=k1,
            stride=s1,
            padding=(0, 0),
            bias=False,
            device=device,
            dtype=dtype,
        )

        f_out_1 = (f_out_0 + 2 - k1[1]) // s1[1] + 1

        self.norm_1 = CumulativeGroupNorm(
            num_channels=ch1,
            feature_dims=(f_out_1,),
            eps=group_norm_eps,
            device=device,
            dtype=dtype,
        )

        flattened_size = ch1 * f_out_1

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
        :returns: Subsampled features ``[B, T/4, H]`` where H=1536.
        """
        batch_size = features.size(0)

        # [B, T, F] -> [B, 1, T, F]
        x = features.unsqueeze(1)

        # Block 0
        x = F.pad(x, self._pad_0).to(self.conv_0.weight.dtype)
        x = self.conv_0(x)  # [B, C0, T', F']
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, T', F', C0]
        x = self.norm_0(x)
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, C0, T', F']
        x = self.activation(x)

        # Block 1
        x = F.pad(x, self._pad_1).to(self.conv_1.weight.dtype)
        x = self.conv_1(x)  # [B, C1, T'', F'']
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, T'', F'', C1]
        x = self.norm_1(x)
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, C1, T'', F'']
        x = self.activation(x)

        # Flatten and project
        x = x.permute(0, 2, 3, 1)  # [B, T'', F'', C1]
        x = x.reshape(batch_size, x.size(1), -1)
        x = self.proj(x)

        return x
