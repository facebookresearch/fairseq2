# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import final

from torch import Tensor
from torch.nn import Module, ModuleList, SiLU
from typing_extensions import override

from fairseq2.data_type import DataType
from fairseq2.device import Device
from fairseq2.models.conformer import ConformerBlock, ConformerConvolution
from fairseq2.models.gemma3n.config import Gemma3nAudioConfig
from fairseq2.models.gemma3n.conformer_sdpa import Gemma3nConformerSDPA
from fairseq2.models.transformer import (
    AttentionBiasCache,
    StandardFeedForwardNetwork,
    StandardMultiheadAttention,
)
from fairseq2.nn import BatchLayout, RMSNorm


@final
class Gemma3nConformerEncoder(Module):
    """Gemma3n audio encoder using USM Conformer architecture.

    Stacks 12 conformer blocks with Shaw relative position embeddings,
    chunked local attention, per-dimension scaling, and macaron-style FFN.
    Applies 4x downsampling after conformer processing.
    """

    layers: ModuleList
    reduction_factor: int

    def __init__(
        self,
        config: Gemma3nAudioConfig,
        *,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        """
        :param config: Audio tower configuration.
        """
        super().__init__()

        self.reduction_factor = config.conf_reduction_factor

        layers = []

        for _ in range(config.conf_num_hidden_layers):
            layer = self._build_conformer_block(config, device=device, dtype=dtype)
            layers.append(layer)

        self.layers = ModuleList(layers)

    def _build_conformer_block(
        self,
        config: Gemma3nAudioConfig,
        *,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> ConformerBlock:
        """Build a single conformer block with Gemma3n-specific components."""
        inner_dim = config.hidden_size * 4

        ffn1 = StandardFeedForwardNetwork(
            model_dim=config.hidden_size,
            inner_dim=inner_dim,
            bias=False,
            inner_activation=SiLU(),
            device=device,
            dtype=dtype,
        )

        ffn1_layer_norm = RMSNorm(
            config.hidden_size,
            bias=True,
            eps=config.rms_norm_eps,
            device=device,
            dtype=dtype,
        )

        sdpa = Gemma3nConformerSDPA(
            model_dim=config.hidden_size,
            num_heads=config.conf_num_attention_heads,
            max_left_rel_pos=config.conf_attention_context_left,
            max_right_rel_pos=config.conf_attention_context_right,
            chunk_size=config.conf_attention_chunk_size,
            left_context=config.conf_attention_context_left,
            right_context=config.conf_attention_context_right,
            logit_cap=config.conf_attention_logit_cap,
            device=device,
            dtype=dtype,
        )

        self_attn = StandardMultiheadAttention(
            model_dim=config.hidden_size,
            num_heads=config.conf_num_attention_heads,
            sdpa=sdpa,
            bias=False,
            device=device,
            dtype=dtype,
        )

        self_attn_layer_norm = RMSNorm(
            config.hidden_size,
            bias=True,
            eps=config.rms_norm_eps,
            device=device,
            dtype=dtype,
        )

        conv = ConformerConvolution(
            model_dim=config.hidden_size,
            depthwise_kernel_size=config.conf_conv_kernel_size,
            norm_type="layer_norm",
            device=device,
            dtype=dtype,
        )

        conv_layer_norm = RMSNorm(
            config.hidden_size,
            bias=True,
            eps=config.rms_norm_eps,
            device=device,
            dtype=dtype,
        )

        ffn2 = StandardFeedForwardNetwork(
            model_dim=config.hidden_size,
            inner_dim=inner_dim,
            bias=False,
            inner_activation=SiLU(),
            device=device,
            dtype=dtype,
        )

        ffn2_layer_norm = RMSNorm(
            config.hidden_size,
            bias=True,
            eps=config.rms_norm_eps,
            device=device,
            dtype=dtype,
        )

        layer_norm = RMSNorm(
            config.hidden_size,
            bias=True,
            eps=config.rms_norm_eps,
            device=device,
            dtype=dtype,
        )

        return ConformerBlock(
            ffn1_layer_norm=ffn1_layer_norm,
            ffn1=ffn1,
            self_attn_layer_norm=self_attn_layer_norm,
            self_attn=self_attn,
            conv_layer_norm=conv_layer_norm,
            conv=conv,
            ffn2_layer_norm=ffn2_layer_norm,
            ffn2=ffn2,
            layer_norm=layer_norm,
            dropout_p=0.0,
            device=device,
            dtype=dtype,
        )

    @override
    def forward(
        self, seqs: Tensor, seqs_layout: BatchLayout, mask: Tensor | None = None
    ) -> Tensor:
        """
        :param seqs: Audio features. *Shape:* :math:`(N,T,H)` where H=1536.
        :param seqs_layout: Layout information for the sequences.
        :param mask: Optional mask where True=masked (invalid). *Shape:* :math:`(N,T)`.
        :returns: Encoded audio features. *Shape:* :math:`(N,T/R,H)` where R=reduction_factor.
        """
        bias_cache = AttentionBiasCache()

        for layer in self.layers:
            seqs = layer(seqs, seqs_layout, bias_cache)

        # Apply conformer reduction (4x downsampling via strided slicing)
        if self.reduction_factor > 1:
            seqs = seqs[:, :: self.reduction_factor]
            if mask is not None:
                mask = mask[:, :: self.reduction_factor]

        # Apply masking: set masked positions to zero (HF compatibility)
        if mask is not None:
            seqs = seqs.masked_fill(mask.unsqueeze(-1), 0.0)

        return seqs
