# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

GEMMA3N_FAMILY: Final = "gemma3n"


@dataclass(kw_only=True)
class Gemma3nConfig:
    """Holds the configuration of a Gemma3n model.

    The default values correspond to the E4B architecture as described in the
    Gemma 3 Technical Report (https://arxiv.org/abs/2503.19786).
    """

    model_dim: int = 2048
    """The dimensionality of the model."""

    max_seq_len: int = 32_768
    """The maximum sequence length."""

    vocab_size: int = 262_400
    """The size of the vocabulary."""

    pad_idx: int | None = 0
    """The index of the PAD symbol in the vocabulary."""

    tied_embeddings: bool = True
    """If ``True``, ties the embedding table and the output projection layer."""

    num_layers: int = 35
    """The number of decoder layers."""

    num_attn_heads: int = 8
    """The number of attention heads in decoder layers."""

    num_key_value_heads: int = 2
    """The number of key/value heads for Grouped Query Attention."""

    head_dim: int = 256
    """The dimensionality of attention heads."""

    ffn_inner_dim: int = 16_384
    """The dimensionality of inner projection layers in feed-forward networks."""

    altup_hidden_dim: int = 5376
    """The dimensionality of the AltUp FFN inner projection for local layers."""

    sliding_window: int = 512
    """The sliding window size for local attention layers."""

    rope_theta: float = 10_000.0
    """The RoPE theta for local/standard frequencies."""

    rope_theta_global: float = 1_000_000.0
    """The RoPE theta for global attention frequencies."""

    final_logit_soft_cap: float = 30.0
    """Soft-capping value for attention logits."""

    num_kv_shared_layers: int = 15
    """The number of layers that share KV cache values."""

    laurel_rank: int = 64
    """The rank for LAuReL low-rank residual connections."""

    altup_num_inputs: int = 4
    """The number of predictions that AltUp should make given the input sequence."""

    altup_active_idx: int = 0
    """The index of the prediction from which AltUp will compute additional predictions or correct."""

    altup_coef_clip: float = 120.0
    """The maximum amplitude of an AltUp prediction or correction coefficient weight."""

    altup_correct_scale: bool = True
    """If True, apply the AltUp.correct_output_scale to the corrected prediction."""

    vocab_size_per_layer_input: int = 262_144
    """Vocabulary size of the per-layer text embeddings that augment the standard embeddings."""

    hidden_size_per_layer_input: int = 256
    """Dimension of the hidden representations for per-layer embeddings."""

    rms_norm_eps: float = 1e-6
    """The epsilon value for RMSNorm."""

    dropout_p: float = 0.0
    """The dropout probability on outputs of Transformer layers."""

    init_std: float | None = 0.02
    """The standard deviation to initialize input embeddings and projection weights."""


def is_global_layer(layer_idx: int, num_layers: int = 35) -> bool:
    """Determine if a layer uses global attention.

    Gemma3n uses a 4:1 local:global ratio. Global layers occur every 5th layer
    (indices 4, 9, 14, ...) with the last layer always being global.

    Args:
        layer_idx: The zero-based index of the layer.
        num_layers: The total number of layers.

    Returns:
        True if the layer should use global attention, False for local.
    """
    # Last layer is always global
    if layer_idx == num_layers - 1:
        return True

    # Every 5th layer is global (0-indexed: 4, 9, 14, 19, ...)
    return (layer_idx + 1) % 5 == 0


def get_gemma3n_e2b_config() -> Gemma3nConfig:
    """Get configuration for Gemma3n E2B (2B effective parameters)."""
    return Gemma3nConfig()


def get_gemma3n_e4b_config() -> Gemma3nConfig:
    """Get configuration for Gemma3n E4B (4B effective parameters)."""
    return Gemma3nConfig()
