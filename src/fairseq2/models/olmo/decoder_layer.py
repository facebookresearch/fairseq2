# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""OLMO-specific decoder layer with custom OLMO Post-Norm order."""

from __future__ import annotations

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
from fairseq2.nn import (
    AdditiveResidualConnect,
    BatchLayout,
    IncrementalStateBag,
    LayerNorm,
    ResidualConnect,
)


class OLMOTransformerLMDecoderLayer(TransformerLMDecoderLayer):
    """OLMO Transformer Decoder Layer with custom Post-Norm order.

    The key difference from StandardTransformerLMDecoderLayer is the order
    of operations in Post-Norm:
    - Standard Post-Norm: Attention/FFN -> Add Residual -> Norm
    - OLMO Post-Norm: Attention/FFN -> Norm -> Add Residual
    - Pre-Norm (e.g. LLAMA): Norm -> Attention/FFN -> Add Residual

    This matches the HuggingFace OLMO implementation where normalization
    is applied to the output before adding the residual connection.
    """

    def __init__(
        self,
        self_attn: MultiheadAttention,
        self_attn_layer_norm: LayerNorm,
        ffn: FeedForwardNetwork,
        ffn_layer_norm: LayerNorm,
        *,
        self_attn_residual: ResidualConnect | None = None,
        ffn_residual: ResidualConnect | None = None,
        dropout_p: float = 0.0,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        """Initialize OLMO2 Transformer Decoder Layer.

        :param self_attn: The self attention layer.
        :param self_attn_layer_norm: Layer norm for self attention.
        :param ffn: The feed-forward network.
        :param ffn_layer_norm: Layer norm for FFN.
        :param self_attn_residual: The residual connection for self attention.
        :param ffn_residual: The residual connection for FFN.
        :param dropout_p: The dropout probability.
        """
        super().__init__()

        self.self_attn = self_attn
        self.self_attn_layer_norm = self_attn_layer_norm

        if dropout_p > 0.0:
            self_attn_dropout = Dropout(dropout_p)
        else:
            self_attn_dropout = None

        self.self_attn_dropout: Dropout | None
        self.register_module("self_attn_dropout", self_attn_dropout)

        if self_attn_residual is None:
            self_attn_residual = AdditiveResidualConnect()

        self.self_attn_residual = self_attn_residual

        self.ffn = ffn
        self.ffn_layer_norm = ffn_layer_norm

        if dropout_p > 0.0:
            ffn_dropout = Dropout(dropout_p)
        else:
            ffn_dropout = None

        self.ffn_dropout: Dropout | None
        self.register_module("ffn_dropout", ffn_dropout)

        if ffn_residual is None:
            ffn_residual = AdditiveResidualConnect()

        self.ffn_residual = ffn_residual

    @override
    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        attn_bias_cache: AttentionBiasCache,
        *,
        state_bag: IncrementalStateBag | None = None,
    ) -> Tensor:
        seqs = self._forward_self_attn(seqs, seqs_layout, attn_bias_cache, state_bag)
        seqs = self._forward_ffn(seqs)
        return seqs

    def _forward_self_attn(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        attn_bias_cache: AttentionBiasCache,
        state_bag: IncrementalStateBag | None,
    ) -> Tensor:
        residual = seqs

        # Attention
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

        # OLMO2-specific order: Norm THEN add residual
        seqs = self.self_attn_layer_norm(seqs)
        seqs = self.self_attn_residual(seqs, residual)

        return seqs

    def _forward_ffn(self, seqs: Tensor) -> Tensor:
        residual = seqs

        # FFN
        seqs = self.ffn(seqs)

        if self.ffn_dropout is not None:
            seqs = self.ffn_dropout(seqs)

        # OLMO2-specific order: Norm THEN add residual
        seqs = self.ffn_layer_norm(seqs)
        seqs = self.ffn_residual(seqs, residual)

        return seqs
