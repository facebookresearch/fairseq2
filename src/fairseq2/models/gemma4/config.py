# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

from fairseq2.models.gemma3n.kv_projection import KVProjectionRole
from fairseq2.models.gemma4.audio.config import Gemma4AudioConfig
from fairseq2.runtime.config_registry import ConfigRegistrar
from fairseq2.runtime.dependency import DependencyContainer

GEMMA4_FAMILY: Final = "gemma4"


@dataclass(kw_only=True)
class Gemma4Config:
    """Holds the configuration of a Gemma 4 model.

    The default values correspond to the E4B architecture.
    """

    model_dim: int = 2560
    """The dimensionality of the model (hidden_size)."""

    max_seq_len: int = 131_072
    """The maximum sequence length."""

    vocab_size: int = 262_144
    """The size of the vocabulary."""

    pad_idx: int | None = 0
    """The index of the PAD symbol in the vocabulary."""

    tied_embeddings: bool = True
    """If ``True``, ties the embedding table and the output projection layer."""

    num_layers: int = 42
    """The number of decoder layers."""

    num_attn_heads: int = 8
    """The number of attention heads in decoder layers."""

    num_key_value_heads: int = 2
    """The number of key/value heads for Grouped Query Attention (sliding layers)."""

    head_dim: int = 256
    """The dimensionality of attention heads for sliding attention layers."""

    global_head_dim: int = 512
    """The dimensionality of attention heads for full (global) attention layers."""

    num_global_key_value_heads: int | None = None
    """Number of key/value heads for global attention layers.
    If None, defaults to num_key_value_heads."""

    ffn_inner_dim: int = 10_240
    """The dimensionality of inner projection layers in feed-forward networks."""

    sliding_window: int = 512
    """The sliding window size for local attention layers."""

    rope_theta: float = 10_000.0
    """The RoPE theta for sliding (local) attention layers."""

    rope_theta_global: float = 1_000_000.0
    """The RoPE theta for global (full) attention layers."""

    partial_rotary_factor: float = 0.25
    """Fraction of head_dim that gets rotary encoding in full attention layers."""

    attention_k_eq_v: bool = False
    """If True, key projection output is reused as value (no separate v_proj)
    for full attention layers."""

    num_kv_shared_layers: int = 18
    """Number of consecutive decoder layers at the end that share KV projections."""

    use_double_wide_mlp: bool = False
    """If True, KV-shared layers use 2x intermediate_size in the MLP."""

    enable_moe: bool = False
    """If True, enable Mixture-of-Experts blocks parallel to dense MLP."""

    num_experts: int | None = None
    """Number of MoE experts per layer (only used when enable_moe=True)."""

    top_k_experts: int | None = None
    """Number of experts activated per token (only used when enable_moe=True)."""

    moe_intermediate_size: int | None = None
    """Intermediate size of each expert's FFN (only used when enable_moe=True)."""

    final_logit_soft_cap: float | None = 30.0
    """Soft-capping value for final logits. None to disable."""

    vocab_size_per_layer_input: int = 262_144
    """Vocabulary size of the per-layer text embeddings (PLE)."""

    hidden_size_per_layer_input: int = 256
    """Dimension of the hidden representations for per-layer embeddings.
    Set to 0 to disable PLE."""

    layer_types: list[str] = field(default_factory=list)
    """Per-layer attention type list. If empty, computed from 5:1 pattern."""

    rms_norm_eps: float = 1e-6
    """The epsilon value for RMSNorm."""

    dropout_p: float = 0.0
    """The dropout probability on outputs of Transformer layers."""

    init_std: float | None = 0.02
    """The standard deviation to initialize input embeddings and projection weights."""

    hidden_activation: str = "gelu_pytorch_tanh"
    """The activation function used in FFN and PLE."""

    audio_config: Gemma4AudioConfig | None = None
    """Audio tower configuration. None means text-only model."""

    audio_token_id: int = 258_881
    """Token ID used as placeholder for audio embeddings."""

    @property
    def ple_hidden_dim(self) -> int:
        """Alias for hidden_size_per_layer_input."""
        return self.hidden_size_per_layer_input

    @property
    def has_ple(self) -> bool:
        """Whether Per-Layer Embeddings are enabled."""
        return self.hidden_size_per_layer_input > 0

    @property
    def final_logit_softcapping(self) -> float | None:
        """Alias for final_logit_soft_cap."""
        return self.final_logit_soft_cap

    def __post_init__(self) -> None:
        if not self.layer_types:
            self.layer_types = _compute_layer_types(self.num_layers)

        if self.num_global_key_value_heads is None:
            self.num_global_key_value_heads = self.num_key_value_heads


def _compute_layer_types(num_layers: int) -> list[str]:
    """Compute the 5:1 sliding:full attention pattern.

    Every 6th layer (1-indexed) is full_attention, with the last layer
    always being full_attention.
    """
    sliding_window_pattern = 6  # 5 sliding : 1 full
    layer_types = [
        (
            "sliding_attention"
            if bool((i + 1) % sliding_window_pattern)
            else "full_attention"
        )
        for i in range(num_layers)
    ]
    # Last layer must be full_attention
    if layer_types and layer_types[-1] != "full_attention":
        layer_types[-1] = "full_attention"
    return layer_types


def get_kv_projection_role(
    layer_idx: int,
    layer_type: str,
    num_layers: int,
    num_kv_shared_layers: int,
    layer_types: list[str],
) -> KVProjectionRole:
    """Determine KV projection sharing role for a layer.

    :param layer_idx: Zero-based layer index.
    :param layer_type: "sliding_attention" or "full_attention".
    :param num_layers: Total number of layers.
    :param num_kv_shared_layers: Number of layers that consume shared K/V.
    :param layer_types: Per-layer attention type list.
    :returns: KVProjectionRole indicating this layer's role in KV sharing.
    """
    if num_kv_shared_layers <= 0:
        return KVProjectionRole.NONE

    first_shared_idx = num_layers - num_kv_shared_layers

    # All layers from first_shared_idx onwards are consumers
    if layer_idx >= first_shared_idx:
        return KVProjectionRole.CONSUMER

    # Check if this is the last layer of its type before sharing starts.
    # That layer becomes the source for all consumers of the same type.
    for idx in range(layer_idx + 1, first_shared_idx):
        if layer_types[idx] == layer_type:
            return KVProjectionRole.NONE  # Found a later layer of same type

    return KVProjectionRole.SOURCE  # This is the last of its type before sharing


def register_gemma4_configs(container: DependencyContainer) -> None:
    """Register Gemma4 model configurations."""
    arch = ConfigRegistrar(container, Gemma4Config)

    @arch("e4b")
    def _e4b() -> Gemma4Config:
        return get_gemma4_e4b_config()

    @arch("e4b_it")
    def _e4b_it() -> Gemma4Config:
        return get_gemma4_e4b_config()

    @arch("31b")
    def _31b() -> Gemma4Config:
        return get_gemma4_31b_config()

    @arch("31b_it")
    def _31b_it() -> Gemma4Config:
        return get_gemma4_31b_config()

    @arch("26b_a4b")
    def _26b_a4b() -> Gemma4Config:
        return get_gemma4_26b_a4b_config()

    @arch("26b_a4b_it")
    def _26b_a4b_it() -> Gemma4Config:
        return get_gemma4_26b_a4b_config()

    @arch("e2b")
    def _e2b() -> Gemma4Config:
        return get_gemma4_e2b_config()

    @arch("e2b_it")
    def _e2b_it() -> Gemma4Config:
        return get_gemma4_e2b_config()

    @arch("12b")
    def _12b() -> Gemma4Config:
        return get_gemma4_12b_config()

    @arch("12b_it")
    def _12b_it() -> Gemma4Config:
        return get_gemma4_12b_config()

    @arch("12b_audio")
    def _12b_audio() -> Gemma4Config:
        return get_gemma4_12b_audio_config()

    @arch("12b_it_audio")
    def _12b_it_audio() -> Gemma4Config:
        return get_gemma4_12b_audio_config()


def get_gemma4_e2b_config() -> Gemma4Config:
    """Get configuration for Gemma4 E2B (small dense, on-device).

    E2B uses a 4:1 sliding:full attention pattern (every 5th layer is full)
    instead of E4B's 5:1 (every 6th). It also uses ``use_double_wide_mlp``
    to compensate for parameter savings from aggressive KV sharing (20 layers).
    """
    # 4:1 pattern: full attention at indices 4,9,14,19,24,29,34
    layer_types = [
        "full_attention" if (i + 1) % 5 == 0 else "sliding_attention" for i in range(35)
    ]

    return Gemma4Config(
        model_dim=1536,
        max_seq_len=131_072,
        num_layers=35,
        num_attn_heads=8,
        num_key_value_heads=1,
        head_dim=256,
        global_head_dim=512,
        num_global_key_value_heads=None,  # defaults to num_key_value_heads=1
        ffn_inner_dim=6_144,
        sliding_window=512,
        attention_k_eq_v=False,
        num_kv_shared_layers=20,
        use_double_wide_mlp=True,
        hidden_size_per_layer_input=256,  # PLE enabled
        final_logit_soft_cap=30.0,
        layer_types=layer_types,
    )


def get_gemma4_e4b_config() -> Gemma4Config:
    """Get configuration for Gemma4 E4B."""
    return Gemma4Config(
        model_dim=2560,
        max_seq_len=131_072,
        num_layers=42,
        num_attn_heads=8,
        num_key_value_heads=2,
        head_dim=256,
        global_head_dim=512,
        num_global_key_value_heads=None,  # defaults to num_key_value_heads=2
        ffn_inner_dim=10_240,
        sliding_window=512,
        attention_k_eq_v=False,
        num_kv_shared_layers=18,
        hidden_size_per_layer_input=256,  # PLE enabled
        final_logit_soft_cap=30.0,
    )


def get_gemma4_31b_config() -> Gemma4Config:
    """Get configuration for Gemma4 31B (dense)."""
    return Gemma4Config(
        model_dim=5376,
        max_seq_len=262_144,
        num_layers=60,
        num_attn_heads=32,
        num_key_value_heads=16,
        head_dim=256,
        global_head_dim=512,
        num_global_key_value_heads=4,
        ffn_inner_dim=21_504,
        sliding_window=1024,
        attention_k_eq_v=True,
        num_kv_shared_layers=0,
        hidden_size_per_layer_input=0,  # PLE disabled
        final_logit_soft_cap=30.0,
    )


def get_gemma4_26b_a4b_config() -> Gemma4Config:
    """Get configuration for Gemma4 26B-A4B (MoE variant)."""
    return Gemma4Config(
        model_dim=2816,
        max_seq_len=262_144,
        num_layers=30,
        num_attn_heads=16,
        num_key_value_heads=8,
        head_dim=256,
        global_head_dim=512,
        num_global_key_value_heads=2,
        ffn_inner_dim=2112,
        sliding_window=1024,
        attention_k_eq_v=True,
        num_kv_shared_layers=0,
        hidden_size_per_layer_input=0,  # PLE disabled
        enable_moe=True,
        num_experts=128,
        top_k_experts=8,
        moe_intermediate_size=704,
        final_logit_soft_cap=30.0,
    )


def get_gemma4_unified_audio_config() -> Gemma4AudioConfig:
    """Audio config for the Gemma 4 Unified family (12B+).

    Linear (tower-free) pipeline: raw 16 kHz waveform is chunked into frames
    of ``audio_samples_per_token`` = 640 samples (40 ms each), then projected
    to the text model dim through RMSNorm + Linear. No mel-spectrogram, no
    Conformer.

    Matches HF's ``Gemma4UnifiedAudioConfig`` (model_type=gemma4_unified_audio)
    where ``audio_embed_dim = audio_samples_per_token = output_proj_dims = 640``.
    """
    return Gemma4AudioConfig(
        audio_mode="linear",
        # In linear mode, only output_proj_dims (= 640 raw samples per token)
        # and rms_norm_eps are read. Other Conformer fields default values
        # are unused.
        output_proj_dims=640,
        rms_norm_eps=1e-6,
    )


def get_gemma4_12b_config() -> Gemma4Config:
    """Get configuration for Gemma4 12B (Unified family, dense).

    First member of HF ``gemma4_unified`` model_type (released 2026-05-23).
    The text decoder is a dense Gemma 4 model that reuses the same attention,
    decoder, and frontend code paths as the existing 31B dense variant.
    Distinguishing values vs the existing dense archs:

    * ``num_global_key_value_heads = 1`` — multi-query (MQA) global attention.
      Previously the dense archs only used 2 (26B-A4B) and 4 (31B).
    * ``attention_k_eq_v = True`` — keys reused as values in global layers.
    * ``hidden_size_per_layer_input = 0`` — no PLE.
    * ``num_kv_shared_layers = 0`` — no KV sharing.
    * 48 layers, 5:1 sliding:full pattern (40 sliding + 8 full).

    The 12B Unified checkpoint also ships an ``embed_audio`` projection
    (``[3840, 640]``) and a ``vision_embedder`` pipeline (LN + Dense + LN +
    factorized 2D positional embedding + RMSNorm + Linear), but those are
    not required for text-only logit parity or downstream text evaluation.
    Multimodal embedders are intentionally not registered here — when
    ``audio_config`` is ``None`` (the default), :func:`convert_gemma4_state_dict`
    filters multimodal keys (audio_tower, embed_audio, vision_tower,
    embed_vision, vision_embedder, multi_modal_projector).
    """
    return Gemma4Config(
        model_dim=3840,
        max_seq_len=262_144,
        num_layers=48,
        num_attn_heads=16,
        num_key_value_heads=8,
        head_dim=256,
        global_head_dim=512,
        num_global_key_value_heads=1,
        ffn_inner_dim=15_360,
        sliding_window=1024,
        attention_k_eq_v=True,
        num_kv_shared_layers=0,
        hidden_size_per_layer_input=0,  # PLE disabled (Unified family has no PLE)
        final_logit_soft_cap=30.0,
    )


def get_gemma4_12b_audio_config() -> Gemma4Config:
    """Get configuration for Gemma4 12B (Unified family) WITH the audio
    embedder enabled.

    Identical to :func:`get_gemma4_12b_config` except that ``audio_config`` is
    set to :func:`get_gemma4_unified_audio_config` (linear mode, no tower).
    Use this arch when you want to consume audio inputs through the
    fairseq2 inference path (audio+text -> text).

    The text-only ``12b`` / ``12b_it`` archs are unchanged and remain the
    canonical entry point for logit parity, MMLU, SFT — keeping the audio
    embedder out of those configs avoids loading unused parameters and
    preserves the converter's multimodal filter.
    """
    cfg = get_gemma4_12b_config()
    cfg.audio_config = get_gemma4_unified_audio_config()
    return cfg
