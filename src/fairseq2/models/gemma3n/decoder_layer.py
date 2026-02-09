# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import TYPE_CHECKING, final

from torch import Tensor
from torch.nn import Dropout
from typing_extensions import override

from fairseq2.data_type import DataType
from fairseq2.device import Device
from fairseq2.models.transformer import (
    AttentionBiasCache,
    FeedForwardNetwork,
    MultiheadAttention,
)
from fairseq2.models.transformer_lm import TransformerLMDecoderLayer
from fairseq2.nn import BatchLayout, IncrementalStateBag, LayerNorm, ResidualConnect


@final
class Gemma3nDecoderLayer(TransformerLMDecoderLayer):
    """Gemma3n decoder layer with LAuReL residuals and optional PLE augmentation."""

    self_attn: MultiheadAttention
    ffn: FeedForwardNetwork
    input_layernorm: LayerNorm
    post_attention_layernorm: LayerNorm
    self_attn_residual: ResidualConnect
    ffn_residual: ResidualConnect
    self_attn_dropout: Dropout | None
    ffn_dropout: Dropout | None
    pre_feedforward_layernorm: LayerNorm | None
    post_feedforward_layernorm: LayerNorm | None
    ple: Module | None

    def __init__(
        self,
        self_attn: MultiheadAttention,
        ffn: FeedForwardNetwork,
        *,
        input_layernorm: LayerNorm,
        post_attention_layernorm: LayerNorm,
        self_attn_residual: ResidualConnect,
        ffn_residual: ResidualConnect,
        pre_feedforward_layernorm: LayerNorm | None = None,
        post_feedforward_layernorm: LayerNorm | None = None,
        ple: Module | None = None,
        dropout_p: float = 0.0,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        """
        :param self_attn: Self-attention layer with QK normalization.
        :param ffn: Feed-forward network (AltUp or standard GLU).
        :param input_layernorm: Pre-attention normalization.
        :param post_attention_layernorm: Pre-FFN normalization.
        :param self_attn_residual: LAuReL residual with post-LAuReL normalization.
        :param ffn_residual: Residual connection for FFN.
        :param pre_feedforward_layernorm: Additional pre-FFN normalization (optional).
        :param post_feedforward_layernorm: Post-FFN normalization (optional).
        :param ple: Per-Layer Embedding augmentation (optional).
        :param dropout_p: Dropout probability.
        """
        super().__init__()

        self.self_attn = self_attn
        self.ffn = ffn

        self.input_layernorm = input_layernorm
        self.post_attention_layernorm = post_attention_layernorm
        self.self_attn_residual = self_attn_residual
        self.ffn_residual = ffn_residual

        if pre_feedforward_layernorm is not None:
            self.register_module("pre_feedforward_layernorm", pre_feedforward_layernorm)
        else:
            self.pre_feedforward_layernorm = None

        if post_feedforward_layernorm is not None:
            self.register_module("post_feedforward_layernorm", post_feedforward_layernorm)
        else:
            self.post_feedforward_layernorm = None

        if ple is not None:
            self.register_module("ple", ple)
        else:
            self.ple = None

        if dropout_p > 0.0:
            self.self_attn_dropout = Dropout(dropout_p)
            self.ffn_dropout = Dropout(dropout_p)
        else:
            self.self_attn_dropout = None
            self.ffn_dropout = None

    @override
    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        attn_bias_cache: AttentionBiasCache,
        *,
        state_bag: IncrementalStateBag | None = None,
    ) -> Tensor:
        # Attention block: pre-norm + self-attention + LAuReL residual
        residual = seqs
        seqs = self.input_layernorm(seqs)
        seqs = self.self_attn(
            seqs,
            seqs_layout,
            keys=seqs,
            keys_layout=seqs_layout,
            values=seqs,
            bias_cache=attn_bias_cache,
            state_bag=state_bag,
        )

        if self.self_attn_dropout is not None:
            seqs = self.self_attn_dropout(seqs)

        seqs = self.self_attn_residual(seqs, residual)

        # FFN block: pre-norm(s) + FFN + post-norm (optional) + residual
        residual = seqs
        seqs = self.post_attention_layernorm(seqs)

        if self.pre_feedforward_layernorm is not None:
            seqs = self.pre_feedforward_layernorm(seqs)

        seqs = self.ffn(seqs)

        if self.post_feedforward_layernorm is not None:
            seqs = self.post_feedforward_layernorm(seqs)

        if self.ffn_dropout is not None:
            seqs = self.ffn_dropout(seqs)

        seqs = self.ffn_residual(seqs, residual)

        return seqs
