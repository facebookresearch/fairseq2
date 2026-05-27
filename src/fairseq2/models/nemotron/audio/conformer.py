# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Conformer blocks and audio tower for the Parakeet audio encoder.

Implements the Macaron-style conformer block used in NVIDIA's FastConformer:
  1. feed_forward1 (×0.5 scaling) → residual
  2. self_attn → residual
  3. conv → residual
  4. feed_forward2 (×0.5 scaling) → residual
  5. layer_norm → output

Reuses fairseq2's ``ConformerConvolution`` (batch_norm, non-causal) and
``StandardFeedForwardNetwork`` (SiLU activation, bias=True).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, final

import torch
from torch import Tensor
from torch.nn import Module, ModuleList, SiLU
from typing_extensions import override

from fairseq2.data_type import DataType
from fairseq2.device import Device
from fairseq2.models.conformer.convolution import ConformerConvolution
from fairseq2.models.nemotron.audio.attention import (
    ParakeetRelativeAttention,
    ParakeetRelativePositionalEncoding,
)
from fairseq2.models.nemotron.audio.subsample import ParakeetSubsamplingConv2D
from fairseq2.models.transformer.ffn import StandardFeedForwardNetwork
from fairseq2.nn import BatchLayout, StandardLayerNorm


@final
class ParakeetConformerBlock(Module):
    """A single Macaron-style conformer block for Parakeet.

    Forward pass:
        1. ``norm_feed_forward1`` → ``feed_forward1`` → ×0.5 → residual
        2. ``norm_self_att`` → ``self_attn`` → residual
        3. ``norm_conv`` → ``conv`` → residual
        4. ``norm_feed_forward2`` → ``feed_forward2`` → ×0.5 → residual
        5. ``norm_out`` → output
    """

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        ffn_dim: int,
        conv_kernel_size: int = 9,
        *,
        head_dim: int | None = None,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        super().__init__()

        # Pre-norms (all LayerNorm with bias=True per Parakeet)
        self.norm_feed_forward1 = StandardLayerNorm(
            model_dim, bias=True, device=device, dtype=dtype
        )
        self.norm_self_att = StandardLayerNorm(
            model_dim, bias=True, device=device, dtype=dtype
        )
        self.norm_conv = StandardLayerNorm(
            model_dim, bias=True, device=device, dtype=dtype
        )
        self.norm_feed_forward2 = StandardLayerNorm(
            model_dim, bias=True, device=device, dtype=dtype
        )
        self.norm_out = StandardLayerNorm(
            model_dim, bias=True, device=device, dtype=dtype
        )

        # Feed-forward networks (SiLU activation, bias=True)
        # Using StandardFeedForwardNetwork: keys are inner_proj/output_proj
        # (HF uses linear1/linear2 — handled by interop key mapping)
        self.feed_forward1 = StandardFeedForwardNetwork(
            model_dim,
            ffn_dim,
            bias=True,
            inner_activation=SiLU(),
            device=device,
            dtype=dtype,
        )
        self.feed_forward2 = StandardFeedForwardNetwork(
            model_dim,
            ffn_dim,
            bias=True,
            inner_activation=SiLU(),
            device=device,
            dtype=dtype,
        )

        # Relative position self-attention
        self.self_attn = ParakeetRelativeAttention(
            model_dim,
            num_heads,
            head_dim=head_dim,
            bias=True,
            device=device,
            dtype=dtype,
        )

        # Conformer convolution (batch_norm, non-causal, SiLU activation)
        self.conv = ConformerConvolution(
            model_dim,
            conv_kernel_size,
            causal_depthwise_conv=False,
            norm_type="batch_norm",
            device=device,
            dtype=dtype,
        )

    def forward(self, x: Tensor, pos_enc: Tensor, seqs_layout: BatchLayout) -> Tensor:
        """Forward pass through the conformer block.

        :param x:
            Input features. *Shape:* ``[B, T, D]``.
        :param pos_enc:
            Position encodings. *Shape:* ``[T, D]``.
        :param seqs_layout:
            Batch layout for the sequences (needed by ConformerConvolution).

        :returns:
            Output features. *Shape:* ``[B, T, D]``.
        """
        # 1. First Macaron FFN (×0.5 residual)
        residual = x
        x = self.norm_feed_forward1(x)
        x = self.feed_forward1(x)
        x = residual + 0.5 * x

        # 2. Self-attention
        residual = x
        x = self.norm_self_att(x)
        x = self.self_attn(x, pos_enc)
        x = residual + x

        # 3. Conformer convolution
        residual = x
        x = self.norm_conv(x)
        x = self.conv(x, seqs_layout)
        x = residual + x

        # 4. Second Macaron FFN (×0.5 residual)
        residual = x
        x = self.norm_feed_forward2(x)
        x = self.feed_forward2(x)
        x = residual + 0.5 * x

        # 5. Final layer norm
        x = self.norm_out(x)

        return x

    if TYPE_CHECKING:
        __call__ = forward


@final
class ParakeetAudioTower(Module):
    """The full Parakeet (FastConformer) audio encoder tower.

    Pipeline:
        mel spectrogram [B, T, mel_bins]
        → Conv2D subsampling [B, T/8, hidden_size]
        → positional encoding
        → N conformer blocks
        → output [B, T/8, hidden_size]
    """

    def __init__(
        self,
        num_mel_bins: int = 128,
        hidden_size: int = 1024,
        num_heads: int = 8,
        num_layers: int = 24,
        ffn_dim: int = 4096,
        conv_kernel_size: int = 9,
        conv_channels: int = 256,
        *,
        head_dim: int | None = None,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size

        # Conv2D subsampling (8x temporal downsampling)
        self.subsampling = ParakeetSubsamplingConv2D(
            num_mel_bins=num_mel_bins,
            hidden_size=hidden_size,
            conv_channels=conv_channels,
            device=device,
            dtype=dtype,
        )

        # Relative positional encoding (non-persistent buffer)
        self.pos_encoding = ParakeetRelativePositionalEncoding(
            hidden_size, device=device, dtype=dtype
        )

        # Stack of conformer blocks
        self.layers = ModuleList(
            [
                ParakeetConformerBlock(
                    hidden_size,
                    num_heads,
                    ffn_dim,
                    conv_kernel_size,
                    head_dim=head_dim,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, mel_features: Tensor) -> Tensor:
        """Encode mel spectrogram features.

        :param mel_features:
            Mel spectrogram features. *Shape:* ``[B, T, num_mel_bins]``.

        :returns:
            Encoded audio features. *Shape:* ``[B, T/8, hidden_size]``.
        """
        # Subsample: [B, T, mel_bins] -> [B, T/8, hidden_size]
        x = self.subsampling(mel_features)

        b, t, _ = x.shape

        # Compute position encoding for subsampled length
        pos_enc = self.pos_encoding(t)

        # Create batch layout for ConformerConvolution
        seqs_layout = BatchLayout(
            (b, t), seq_lens=None, device=x.device
        )

        # Apply conformer blocks
        for layer in self.layers:
            x = layer(x, pos_enc, seqs_layout)

        return x

    if TYPE_CHECKING:
        __call__ = forward

    @override
    def extra_repr(self) -> str:
        """:meta private:"""
        return f"hidden_size={self.hidden_size}, num_layers={len(self.layers)}"
