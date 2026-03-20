# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from fairseq2.models.gemma3n.kv_projection import KVProjectionRole
from fairseq2.runtime.config_registry import ConfigRegistrar
from fairseq2.runtime.dependency import DependencyContainer

GEMMA3N_FAMILY: Final = "gemma3n"


@dataclass(kw_only=True)
class Gemma3nAudioConfig:
    """Holds the configuration of the Gemma3n audio encoder (USM Conformer).

    The default values correspond to the audio tower in the E2B/E4B models.
    """

    vocab_size: int = 128
    """Vocabulary size of audio hard-token embeddings."""

    vocab_offset: int = 262_272
    """Offset for audio tokens in the main tokenizer (text 262,144 + vision 128)."""

    input_feat_size: int = 128
    """Number of channels in each mel-spectrogram frame."""

    hidden_size: int = 1536
    """Dimension of the audio encoder hidden representations."""

    rms_norm_eps: float = 1e-6
    """Epsilon for RMS normalization layers."""

    gradient_clipping: float = 1e10
    """Gradient clipping value for stability."""

    conf_num_hidden_layers: int = 12
    """Number of conformer layers."""

    conf_num_attention_heads: int = 8
    """Number of attention heads in conformer layers."""

    conf_conv_kernel_size: int = 5
    """Kernel size for depthwise convolution in conformer blocks."""

    conf_attention_chunk_size: int = 12
    """Sub-sequence size for chunked local attention."""

    conf_attention_context_left: int = 13
    """Left context size for local attention."""

    conf_attention_context_right: int = 0
    """Right context size for local attention."""

    conf_attention_logit_cap: float = 50.0
    """Logit cap for attention softcapping."""

    conf_reduction_factor: int = 4
    """Reduction factor for subsampling (downsampling ratio)."""

    conf_residual_weight: float = 0.5
    """Residual connection weight (Macaron-style scaling)."""

    sscp_conv_channel_size: tuple[int, int] = (128, 32)
    """Channel sizes for subsample conv projection layers (conv0, conv1)."""

    sscp_conv_group_norm_eps: float = 1e-3
    """Epsilon for group normalization in subsample conv projection."""

    sscp_conv_kernel_size: tuple[tuple[int, int], tuple[int, int]] = (
        (3, 3),
        (3, 3),
    )
    """Kernel sizes (time, freq) for subsample conv projection layers."""

    sscp_conv_stride_size: tuple[tuple[int, int], tuple[int, int]] = (
        (2, 2),
        (2, 2),
    )
    """Stride sizes (time, freq) for subsample conv projection layers."""


def register_gemma3n_configs(container: DependencyContainer) -> None:
    """Register Gemma3n model configurations."""
    arch = ConfigRegistrar(container, Gemma3nConfig)

    @arch("e2b")
    def _e2b() -> Gemma3nConfig:
        return get_gemma3n_e2b_config()

    @arch("e4b")
    def _e4b() -> Gemma3nConfig:
        return get_gemma3n_e4b_config()


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

    audio_config: Gemma3nAudioConfig | None = None
    """Configuration for the audio encoder tower. If None, audio modality is disabled."""

    num_audio_tokens: int = 188
    """Fixed number of <audio> tokens per audio input (30s @ 16kHz with 16x reduction)."""

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

    altup_hidden_dim: int = 16_384
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

    :param layer_idx: The zero-based index of the layer.
    :param num_layers: The total number of layers.
    :returns: True if the layer should use global attention, False for local.
    """
    # Last layer is always global
    if layer_idx == num_layers - 1:
        return True

    # Every 5th layer is global (0-indexed: 4, 9, 14, 19, ...)
    return (layer_idx + 1) % 5 == 0


def get_kv_projection_role(
    layer_idx: int,
    is_global: bool,
    num_layers: int = 30,
    num_kv_shared_layers: int = 10,
) -> KVProjectionRole:
    """Determine KV projection sharing role for a layer.

    :param layer_idx: Zero-based layer index.
    :param is_global: Whether this is a global (full attention) layer.
    :param num_layers: Total number of layers.
    :param num_kv_shared_layers: Number of layers that consume shared K/V.
    :returns: KVProjectionRole indicating this layer's role in KV sharing.
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
        ffn_inner_dim=8192,
        altup_hidden_dim=8192,
        num_kv_shared_layers=10,
        audio_config=Gemma3nAudioConfig(),
    )


def get_gemma3n_e4b_config() -> Gemma3nConfig:
    """Get configuration for Gemma3n E4B (4B effective parameters)."""
    return Gemma3nConfig(
        num_kv_shared_layers=15,
        audio_config=Gemma3nAudioConfig(),
    )
