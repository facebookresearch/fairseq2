# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from fairseq2.models.gemma3n.kv_projection import KVProjectionRole

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

    num_kv_shared_layers: int = 10
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

    @property
    def ple_hidden_dim(self) -> int:
        """Alias for hidden_size_per_layer_input."""
        return self.hidden_size_per_layer_input

    @property
    def vocab_size_per_layer(self) -> int:
        """Alias for vocab_size_per_layer_input."""
        return self.vocab_size_per_layer_input

    @property
    def final_logit_softcapping(self) -> float:
        """Alias for final_logit_soft_cap to match HF naming."""
        return self.final_logit_soft_cap


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


def get_kv_sharing_config(
    layer_idx: int, num_layers: int = 30, num_kv_shared_layers: int = 10
) -> tuple[bool, int | None, bool]:
    """Determine KV sharing configuration for a layer.

    Gemma3n uses KV sharing where the last `num_kv_shared_layers` layers
    (e.g., 15-29) reuse K/V from earlier layers of the same type instead
    of computing their own.

    Args:
        layer_idx: The zero-based index of the layer.
        num_layers: The total number of layers.
        num_kv_shared_layers: The number of layers that share KV.

    Returns:
        A tuple of (is_kv_shared_layer, kv_source_layer_idx, is_kv_source_layer):
        - is_kv_shared_layer: True if this layer retrieves K/V from a source
        - kv_source_layer_idx: Index of the source layer (if shared), else None
        - is_kv_source_layer: True if this layer stores K/V for shared layers
    """
    first_kv_shared_layer_idx = num_layers - num_kv_shared_layers

    # Check if this is a shared layer
    is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx

    if not is_kv_shared_layer:
        # Check if this layer is a source for shared layers
        # A layer is a source if it's the LAST non-shared layer of its type
        layer_is_global = is_global_layer(layer_idx, num_layers)

        # Find the last non-shared layer of this type
        last_of_type = -1
        for idx in range(first_kv_shared_layer_idx):
            if is_global_layer(idx, num_layers) == layer_is_global:
                last_of_type = idx

        # This layer is a source only if it's the last of its type
        is_kv_source_layer = (layer_idx == last_of_type)

        return False, None, is_kv_source_layer
    else:
        # This is a shared layer - find its source
        layer_is_global = is_global_layer(layer_idx, num_layers)

        # Find the last non-shared layer with the same type
        # Search backwards from first_kv_shared_layer_idx - 1 to 0
        for source_idx in range(first_kv_shared_layer_idx - 1, -1, -1):
            if is_global_layer(source_idx, num_layers) == layer_is_global:
                return True, source_idx, False

        # Should never reach here if config is valid
        raise ValueError(
            f"Layer {layer_idx} is a KV shared layer but no source layer "
            f"of the same type found in layers 0-{first_kv_shared_layer_idx-1}"
        )


def get_kv_projection_role(
    layer_idx: int,
    is_global: bool,
    num_layers: int = 30,
    num_kv_shared_layers: int = 10,
) -> KVProjectionRole:
    """Determine KV projection sharing role for a layer.

    Args:
        layer_idx: Zero-based layer index.
        is_global: Whether this is a global (full attention) layer.
        num_layers: Total number of layers.
        num_kv_shared_layers: Number of layers that consume shared K/V.

    Returns:
        KVProjectionRole indicating this layer's role in KV projection sharing.
    """
    first_shared_idx = num_layers - num_kv_shared_layers

    # All layers from first_shared_idx onwards are consumers
    if layer_idx >= first_shared_idx:
        return KVProjectionRole.CONSUMER

    # Check if this is the last layer of its type before sharing starts
    # That layer becomes the source for all consumers of the same type
    for idx in range(layer_idx + 1, first_shared_idx):
        if is_global_layer(idx, num_layers) == is_global:
            return KVProjectionRole.NONE  # Found a later layer of same type

    return KVProjectionRole.SOURCE  # This is the last of its type before sharing


def get_gemma3n_e2b_config() -> Gemma3nConfig:
    """Get configuration for Gemma3n E2B (2B effective parameters)."""
    return Gemma3nConfig(
        num_layers=30,
        ffn_inner_dim=8192,  # E2B uses 8192 for global layers
        altup_hidden_dim=8192,  # E2B uses same FFN dim for local layers
    )


def get_gemma3n_e4b_config() -> Gemma3nConfig:
    """Get configuration for Gemma3n E4B (4B effective parameters)."""
    return Gemma3nConfig()
