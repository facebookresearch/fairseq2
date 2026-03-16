# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from collections.abc import Callable
from typing import final

from torch import Tensor
from torch.nn import Module
from typing_extensions import override

from fairseq2.data_type import DataType
from fairseq2.device import Device
from fairseq2.models.gemma3n.altup import Gemma3nAltUp
from fairseq2.models.gemma3n.kv_projection import KVProjectionRole
from fairseq2.models.transformer import (
    AttentionBiasCache,
    FeedForwardNetwork,
    MultiheadAttention,
)
from fairseq2.nn import BatchLayout, IncrementalStateBag, LayerNorm
from fairseq2.nn.projection import Linear


@final
class Gemma3nLAuReL(Module):
    """Learned Augmented Residual Layer (LAuReL) for Gemma3n.

    LAuReL provides a low-rank learned transformation that augments the residual path.
    Unlike standard residuals, LAuReL learns to enhance the information flow.
    """

    linear_left: Module
    linear_right: Module
    post_laurel_norm: LayerNorm

    def __init__(
        self,
        model_dim: int,
        rank: int,
        *,
        layer_norm: LayerNorm,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        super().__init__()

        self.linear_left = Linear(
            model_dim, rank, bias=False, device=device, dtype=dtype
        )
        self.linear_right = Linear(
            rank, model_dim, bias=False, device=device, dtype=dtype
        )
        self.post_laurel_norm = layer_norm

    @override
    def forward(self, hidden_states: Tensor) -> Tensor:
        """Apply LAuReL transformation to input."""
        laurel_hidden_states = self.linear_left(hidden_states)
        laurel_hidden_states = self.linear_right(laurel_hidden_states)
        normed_laurel_hidden_states = self.post_laurel_norm(laurel_hidden_states)
        return hidden_states + normed_laurel_hidden_states


@final
class Gemma3nDecoderLayer(Module):
    """Gemma3n decoder layer with AltUp, LAuReL, and optional PLE."""

    self_attn: MultiheadAttention
    ffn: FeedForwardNetwork
    input_layernorm: LayerNorm
    post_attention_layernorm: LayerNorm
    pre_feedforward_layernorm: LayerNorm
    post_feedforward_layernorm: LayerNorm
    laurel: Gemma3nLAuReL
    altup: Gemma3nAltUp
    altup_active_idx: int
    altup_correct_scale: bool
    per_layer_input_gate: Module | None
    per_layer_projection: Module | None
    post_per_layer_input_norm: LayerNorm | None
    hidden_activation: Module | None
    layer_idx: int
    is_global: bool
    kv_projection_role: KVProjectionRole

    def __init__(
        self,
        self_attn: MultiheadAttention,
        ffn: FeedForwardNetwork,
        *,
        layer_idx: int,
        is_global: bool,
        kv_projection_role: KVProjectionRole,
        input_layernorm: LayerNorm,
        post_attention_layernorm: LayerNorm,
        pre_feedforward_layernorm: LayerNorm,
        post_feedforward_layernorm: LayerNorm,
        laurel: Gemma3nLAuReL,
        altup: Gemma3nAltUp,
        altup_active_idx: int = 0,
        altup_correct_scale: bool = False,
        per_layer_input_gate: Module | None = None,
        per_layer_projection: Module | None = None,
        post_per_layer_input_norm: LayerNorm | None = None,
        hidden_activation: Module | None = None,
        dropout_p: float = 0.0,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        """
        :param self_attn: Self-attention layer with QK normalization.
        :param ffn: Feed-forward network.
        :param layer_idx: Index of this layer in the decoder.
        :param is_global: Whether this is a global (full) attention layer.
        :param kv_projection_role: Role in KV projection sharing (source/consumer/none).
        :param input_layernorm: Pre-attention normalization.
        :param post_attention_layernorm: Post-attention normalization.
        :param pre_feedforward_layernorm: Pre-FFN normalization.
        :param post_feedforward_layernorm: Post-FFN normalization.
        :param laurel: LAuReL augmented residual module.
        :param altup: AltUp predict/correct module.
        :param altup_active_idx: Index of actively computed AltUp version.
        :param altup_correct_scale: Whether to scale corrected output.
        :param per_layer_input_gate: PLE gating projection.
        :param per_layer_projection: PLE output projection.
        :param post_per_layer_input_norm: PLE post-normalization.
        :param hidden_activation: Activation for PLE gating.
        :param dropout_p: Dropout probability.
        """
        super().__init__()

        self.self_attn = self_attn
        self.ffn = ffn
        self.layer_idx = layer_idx

        self.input_layernorm = input_layernorm
        self.post_attention_layernorm = post_attention_layernorm
        self.pre_feedforward_layernorm = pre_feedforward_layernorm
        self.post_feedforward_layernorm = post_feedforward_layernorm

        self.laurel = laurel
        self.altup = altup
        self.altup_active_idx = altup_active_idx
        self.altup_correct_scale = altup_correct_scale

        # KV projection sharing configuration
        self.is_global = is_global
        self.kv_projection_role = kv_projection_role

        if per_layer_input_gate is not None:
            self.register_module("per_layer_input_gate", per_layer_input_gate)
            self.register_module("per_layer_projection", per_layer_projection)
            self.register_module("post_per_layer_input_norm", post_per_layer_input_norm)
            self.hidden_activation = hidden_activation
        else:
            self.per_layer_input_gate = None
            self.per_layer_projection = None
            self.post_per_layer_input_norm = None
            self.hidden_activation = None

    @override
    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        attn_bias_cache: AttentionBiasCache,
        *,
        per_layer_input: Tensor | None = None,
        state_bag: IncrementalStateBag | None = None,
        pre_computed_kv: tuple[Tensor, Tensor] | None = None,
        kv_storage_callback: Callable[[Tensor, Tensor], None] | None = None,
    ) -> Tensor:
        """Forward pass with AltUp predict/correct pattern and KV sharing.

        :param seqs: 4D input [num_inputs, batch, seq_len, model_dim] or 3D [batch, seq_len, model_dim].
        :param seqs_layout: Batch layout information.
        :param attn_bias_cache: Attention bias cache.
        :param per_layer_input: PLE embeddings for this layer [batch, seq_len, ple_dim].
        :param state_bag: Incremental state for generation.
        :param pre_computed_kv: Pre-computed K/V from a SOURCE layer (for CONSUMERs).
        :param kv_storage_callback: Callback to store K/V (for SOURCEs).
        :returns: 4D output [num_inputs, batch, seq_len, model_dim] or 3D [batch, seq_len, model_dim].
        """
        # AltUp predict step: 4D → 4D predictions
        is_4d = seqs.ndim == 4
        if is_4d:
            predictions = self.altup(seqs)
            active_prediction = predictions[self.altup_active_idx]
        else:
            # No AltUp mode (for compatibility)
            predictions = None
            active_prediction = seqs

        # Pre-normalization
        active_prediction_normed = self.input_layernorm(active_prediction)

        # LAuReL path: parallel augmentation
        laurel_output = self.laurel(active_prediction_normed)

        attn = self.self_attn(
            active_prediction_normed,
            seqs_layout,
            keys=active_prediction_normed,
            keys_layout=seqs_layout,
            values=active_prediction_normed,
            bias_cache=attn_bias_cache,
            state_bag=state_bag,
            pre_computed_kv=pre_computed_kv,
            kv_storage_callback=kv_storage_callback,
        )
        attn = self.post_attention_layernorm(attn)

        # Combine attention + LAuReL with original input (scaled by 1/sqrt(2))
        attn_gated = active_prediction + attn
        attn_laurel = (attn_gated + laurel_output) / math.sqrt(2.0)

        # FFN block
        attn_norm = self.pre_feedforward_layernorm(attn_laurel)
        attn_ffw = self.ffn(attn_norm)
        attn_ffw_norm = self.post_feedforward_layernorm(attn_ffw)
        attn_ffw_laurel_gated = attn_laurel + attn_ffw_norm

        if not is_4d:
            # No AltUp mode
            return attn_ffw_laurel_gated

        # AltUp correct step: update all 4 versions
        assert predictions is not None  # Always set when is_4d=True
        corrected_predictions = self.altup.correct(predictions, attn_ffw_laurel_gated)

        # Extract active version for PLE processing
        first_prediction = corrected_predictions[self.altup_active_idx].clone()
        if self.altup_correct_scale:
            first_prediction = self.altup.scale_corrected_output(first_prediction)

        # PLE augmentation (if enabled)
        if self.per_layer_input_gate is not None and per_layer_input is not None:
            assert self.hidden_activation is not None
            assert self.per_layer_projection is not None
            assert self.post_per_layer_input_norm is not None

            first_prediction = self.per_layer_input_gate(first_prediction)
            first_prediction = self.hidden_activation(first_prediction)
            first_prediction = first_prediction * per_layer_input
            first_prediction = self.per_layer_projection(first_prediction)
            first_prediction = self.post_per_layer_input_norm(first_prediction)

            # Add PLE contribution to non-active versions (versions 1, 2, 3)
            corrected_predictions[1:] += first_prediction

        return corrected_predictions
