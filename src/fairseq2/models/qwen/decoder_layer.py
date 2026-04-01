# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Hybrid decoder layer for Qwen 3.5.

Each layer holds EITHER a :class:`Qwen35Attention` (full attention with output
gating) OR a :class:`GatedDeltaNet` (linear attention), dispatched by
``layer_type``.  The FFN and layer norms are always present.

Attribute names ``self_attn`` / ``linear_attn`` match HuggingFace for clean
interop key mapping.

Reference: HuggingFace ``modeling_qwen3_5.py`` ``Qwen3_5DecoderLayer``
lines 818-870.
"""

from __future__ import annotations

from typing import Final, final

from torch import Tensor
from typing_extensions import override

from fairseq2.models.qwen.attention import Qwen35Attention
from fairseq2.models.qwen.gated_delta_net import GatedDeltaNet
from fairseq2.models.transformer import (
    AttentionBiasCache,
    FeedForwardNetwork,
)
from fairseq2.models.transformer_lm import TransformerLMDecoderLayer
from fairseq2.nn import (
    AdditiveResidualConnect,
    BatchLayout,
    IncrementalStateBag,
    LayerNorm,
    ResidualConnect,
)


@final
class Qwen35DecoderLayer(TransformerLMDecoderLayer):
    """Hybrid decoder layer that dispatches to full or linear attention.

    * ``layer_type == "full_attention"``: uses :attr:`self_attn`
      (:class:`Qwen35Attention`).
    * ``layer_type == "linear_attention"``: uses :attr:`linear_attn`
      (:class:`GatedDeltaNet`).
    """

    layer_type: Final[str]

    def __init__(
        self,
        layer_type: str,
        self_attn: Qwen35Attention | None,
        linear_attn: GatedDeltaNet | None,
        ffn: FeedForwardNetwork,
        self_attn_layer_norm: LayerNorm,
        ffn_layer_norm: LayerNorm,
        *,
        self_attn_residual: ResidualConnect | None = None,
        ffn_residual: ResidualConnect | None = None,
    ) -> None:
        """
        :param layer_type: ``"full_attention"`` or ``"linear_attention"``.
        :param self_attn: Gated full attention module (only for full layers).
        :param linear_attn: GatedDeltaNet module (only for linear layers).
        :param ffn: Feed-forward network (always present).
        :param self_attn_layer_norm: Pre-attention layer norm.
        :param ffn_layer_norm: Pre-FFN layer norm.
        """
        super().__init__()

        self.layer_type = layer_type

        # Register exactly one token mixer — attribute name matters for interop.
        self.self_attn: Qwen35Attention | None
        self.linear_attn: GatedDeltaNet | None

        if layer_type == "full_attention":
            assert self_attn is not None
            self.register_module("self_attn", self_attn)
            self.register_module("linear_attn", None)
        elif layer_type == "linear_attention":
            assert linear_attn is not None
            self.register_module("self_attn", None)
            self.register_module("linear_attn", linear_attn)
        else:
            raise ValueError(
                f"`layer_type` must be 'full_attention' or 'linear_attention', got '{layer_type}'."
            )

        self.self_attn_layer_norm = self_attn_layer_norm
        self.ffn = ffn
        self.ffn_layer_norm = ffn_layer_norm

        if self_attn_residual is None:
            self_attn_residual = AdditiveResidualConnect()
        self.self_attn_residual = self_attn_residual

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
        seqs = self._forward_token_mixer(seqs, seqs_layout, attn_bias_cache, state_bag)
        seqs = self._forward_ffn(seqs)
        return seqs

    def _forward_token_mixer(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        attn_bias_cache: AttentionBiasCache,
        state_bag: IncrementalStateBag | None,
    ) -> Tensor:
        residual = seqs

        seqs = self.self_attn_layer_norm(seqs)

        if self.layer_type == "linear_attention":
            assert self.linear_attn is not None
            seqs = self.linear_attn(seqs, state_bag=state_bag)
        else:
            assert self.self_attn is not None
            seqs = self.self_attn(
                seqs,
                seqs_layout,
                keys=seqs,
                keys_layout=seqs_layout,
                values=seqs,
                bias_cache=attn_bias_cache,
                state_bag=state_bag,
            )

        seqs = self.self_attn_residual(seqs, residual)
        return seqs

    def _forward_ffn(self, seqs: Tensor) -> Tensor:
        residual = seqs

        seqs = self.ffn_layer_norm(seqs)
        seqs = self.ffn(seqs)
        seqs = self.ffn_residual(seqs, residual)

        return seqs

    @override
    def extra_repr(self) -> str:
        return f"layer_type={self.layer_type}"
