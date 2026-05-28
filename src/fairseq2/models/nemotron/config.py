# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Literal

from fairseq2.runtime.config_registry import ConfigRegistrar
from fairseq2.runtime.dependency import DependencyContainer

NEMOTRON_H_FAMILY: Final = "nemotron_h"


@dataclass(kw_only=True)
class CRADIOVisionConfig:
    """Configuration for the C-RADIO (ViT-Huge) vision encoder."""

    hidden_size: int = 1280
    """The hidden size of the ViT blocks."""

    num_attention_heads: int = 16
    """The number of attention heads."""

    head_dim: int = 80
    """The dimensionality of each attention head (hidden_size / num_heads)."""

    num_hidden_layers: int = 32
    """The number of ViT transformer blocks."""

    intermediate_size: int = 5120
    """The intermediate size of the MLP in each ViT block."""

    patch_size: int = 16
    """The size of image patches (patch_size × patch_size)."""

    num_registers: int = 10
    """The number of register (cls) tokens prepended to the sequence."""

    max_grid_size: int = 128
    """The maximum grid size for position embeddings (128×128 = 16384 patches)."""

    image_size: int = 512
    """The default image size (512×512 pixels → 32×32 grid = 1024 patches)."""

    downsample_ratio: float = 0.5
    """Pixel shuffle spatial downsampling ratio (0.5 = merge 2×2 → 4× channels)."""


@dataclass(kw_only=True)
class ParakeetAudioConfig:
    """Configuration for the Parakeet (FastConformer) audio encoder."""

    hidden_size: int = 1024
    """The hidden size of the conformer layers."""

    num_attention_heads: int = 8
    """The number of attention heads."""

    head_dim: int = 128
    """The dimensionality of each attention head."""

    num_hidden_layers: int = 24
    """The number of conformer layers."""

    intermediate_size: int = 4096
    """The intermediate size of the feed-forward networks."""

    conv_kernel_size: int = 9
    """The kernel size for the conformer convolution module."""

    num_mel_bins: int = 128
    """The number of mel spectrogram bins."""

    subsampling_factor: int = 8
    """Temporal downsampling factor (8 = 3 stages of stride-2 convolutions)."""

    subsampling_conv_channels: int = 256
    """The number of channels in subsampling convolutions."""

    ffn_activation: str = "silu"
    """The activation function for feed-forward networks."""

    convolution_bias: bool = False
    """If ``True``, the conformer convolution module uses bias."""

# The 52-layer hybrid pattern for Nemotron-H 30B-A3B
# M = Mamba2 SSM, E = MoE FFN, A = Full Attention (GQA)
_DEFAULT_HYBRID_PATTERN: Final = (
    "M E M E M A E M E M E M A E M E M E M A E M E M E M A "
    "E M E M E M A E M E M E M E M A E M E M E M E M E"
)

BlockType = Literal["mamba", "moe", "attention"]


def parse_hybrid_pattern(pattern_str: str) -> list[BlockType]:
    """Parse a hybrid pattern string into a list of block types.

    Args:
        pattern_str: Space-separated pattern of M (Mamba2), E (MoE), A (Attention).

    Returns:
        List of block type strings.

    Raises:
        ValueError: If the pattern contains invalid characters.
    """
    block_map: dict[str, BlockType] = {
        "M": "mamba",
        "E": "moe",
        "A": "attention",
    }

    tokens = pattern_str.split()
    result: list[BlockType] = []

    for token in tokens:
        if token not in block_map:
            raise ValueError(
                f"Invalid block type '{token}' in hybrid pattern. "
                f"Expected one of: M (Mamba2), E (MoE), A (Attention)."
            )
        result.append(block_map[token])

    return result


@dataclass(kw_only=True)
class NemotronHConfig:
    """Configuration for NemotronH hybrid Mamba2-Transformer MoE models."""

    # === Model dimensions ===
    model_dim: int = 2688
    """The dimensionality of the model (hidden_size)."""

    max_seq_len: int = 262_144
    """The maximum sequence length (max_position_embeddings)."""

    vocab_size: int = 131_072
    """The size of the vocabulary."""

    tied_embeddings: bool = False
    """If ``True``, ties the embedding table and the output projection layer."""

    # === Layer structure ===
    num_layers: int = 52
    """The number of decoder layers."""

    hybrid_override_pattern: str = _DEFAULT_HYBRID_PATTERN
    """Space-separated pattern of block types: M (Mamba2), E (MoE), A (Attention)."""

    # === Attention config (for 'A' layers) ===
    num_attn_heads: int = 32
    """The number of attention query heads."""

    num_key_value_heads: int = 2
    """The number of key/value heads for Grouped Query Attention."""

    attn_head_dim: int = 128
    """The dimensionality of each attention head."""

    rope_theta: float = 10_000.0
    """The coefficient for the Rotary position encoder."""

    # === Mamba2 SSM config (for 'M' layers) ===
    mamba_num_heads: int = 64
    """Number of SSM heads."""

    mamba_head_dim: int = 64
    """Dimensionality per SSM head."""

    ssm_state_size: int = 128
    """The SSM state expansion factor (N in the Mamba paper)."""

    mamba_n_groups: int = 8
    """Number of groups for B and C projections."""

    conv_kernel: int = 4
    """The kernel size of the depthwise conv1d in Mamba."""

    chunk_size: int = 128
    """Chunk size for the chunked scan algorithm."""

    time_step_min: float = 0.001
    """Minimum time step for dt initialization."""

    time_step_max: float = 0.1
    """Maximum time step for dt initialization."""

    use_conv_bias: bool = True
    """If ``True``, the depthwise conv1d uses a bias."""

    mamba_proj_bias: bool = False
    """If ``True``, in_proj and out_proj in Mamba use bias."""

    # === MoE config (for 'E' layers) ===
    num_experts: int = 128
    """The total number of routed experts."""

    num_experts_per_tok: int = 6
    """Top-K experts selected per token."""

    num_shared_experts: int = 1
    """Number of shared experts (always active)."""

    moe_intermediate_size: int = 1856
    """The intermediate size for each routed expert MLP."""

    shared_expert_intermediate_size: int = 3712
    """The intermediate size for the shared expert MLP."""

    routed_scaling_factor: float = 2.5
    """Scaling factor applied to normalized routing weights."""

    moe_n_group: int = 1
    """Number of groups for group-level top-k pre-selection."""

    moe_topk_group: int = 1
    """Number of top groups selected in group-level pre-selection."""

    norm_topk_prob: bool = True
    """If ``True``, normalize the top-K routing probabilities to sum to 1."""

    # === Normalization ===
    rms_norm_eps: float = 1e-5
    """The epsilon value for RMSNorm."""

    # === Dropout ===
    dropout_p: float = 0.0
    """The dropout probability on outputs of Transformer layers."""

    # === Audio (Parakeet) config ===
    audio_config: ParakeetAudioConfig | None = None
    """If not ``None``, the model includes a Parakeet audio encoder."""

    sound_projection_hidden_size: int = 4096
    """The intermediate hidden size of the sound projection MLP."""

    sound_projection_bias: bool = False
    """If ``True``, the sound projection MLP uses bias."""

    sound_context_token_id: int = 27
    """The token ID for audio placeholder tokens (``<so_embedding>``)."""

    # === Vision (C-RADIO) config ===
    vision_config: CRADIOVisionConfig | None = None
    """If not ``None``, the model includes a C-RADIO vision encoder."""

    vision_projection_hidden_size: int = 20480
    """The intermediate hidden size of the vision projection MLP (mlp1)."""

    vision_projection_bias: bool = False
    """If ``True``, the vision projection MLP uses bias."""

    img_context_token_id: int = 18
    """The token ID for image placeholder tokens (``<image>``)."""

    @property
    def layer_types(self) -> list[BlockType]:
        """Parse the hybrid pattern into a list of block types."""
        return parse_hybrid_pattern(self.hybrid_override_pattern)

    @property
    def mamba_intermediate_size(self) -> int:
        """The intermediate size for Mamba2 (num_heads * head_dim)."""
        return self.mamba_num_heads * self.mamba_head_dim

    @property
    def mamba_conv_dim(self) -> int:
        """The dimension of the conv1d input in Mamba2 (intermediate + 2 * n_groups * ssm_state_size)."""
        return self.mamba_intermediate_size + 2 * self.mamba_n_groups * self.ssm_state_size

    @property
    def mamba_projection_size(self) -> int:
        """Total in_proj output size: gate + x_BC + dt."""
        return (
            self.mamba_intermediate_size  # gate
            + self.mamba_conv_dim  # x_BC (= intermediate + 2*n_groups*state_size)
            + self.mamba_num_heads  # dt
        )

    def validate(self) -> None:
        """Validate the configuration."""
        layer_types = self.layer_types

        if len(layer_types) != self.num_layers:
            raise ValueError(
                f"Hybrid pattern length ({len(layer_types)}) does not match "
                f"num_layers ({self.num_layers})."
            )

        # Count layer types
        mamba_count = sum(1 for t in layer_types if t == "mamba")
        moe_count = sum(1 for t in layer_types if t == "moe")
        attn_count = sum(1 for t in layer_types if t == "attention")

        if mamba_count == 0:
            raise ValueError("Hybrid pattern must contain at least one Mamba2 layer.")

        if moe_count == 0:
            raise ValueError("Hybrid pattern must contain at least one MoE layer.")

        if attn_count == 0:
            raise ValueError("Hybrid pattern must contain at least one Attention layer.")


def register_nemotron_h_configs(container: DependencyContainer) -> None:
    arch = ConfigRegistrar(container, NemotronHConfig)

    @arch("nemotron_h_30b_a3b")
    def nemotron_h_30b_a3b() -> NemotronHConfig:
        """NemotronH 30B total / 3B active parameters."""
        config = NemotronHConfig()
        config.validate()
        return config

    @arch("nemotron_h_30b_a3b_audio")
    def nemotron_h_30b_a3b_audio() -> NemotronHConfig:
        """NemotronH 30B with Parakeet audio encoder."""
        config = NemotronHConfig(audio_config=ParakeetAudioConfig())
        config.validate()
        return config

    @arch("nemotron_h_30b_a3b_vision")
    def nemotron_h_30b_a3b_vision() -> NemotronHConfig:
        """NemotronH 30B with C-RADIO vision encoder."""
        config = NemotronHConfig(vision_config=CRADIOVisionConfig())
        config.validate()
        return config

    @arch("nemotron_h_30b_a3b_omni")
    def nemotron_h_30b_a3b_omni() -> NemotronHConfig:
        """NemotronH 30B with both vision and audio encoders."""
        config = NemotronHConfig(
            vision_config=CRADIOVisionConfig(),
            audio_config=ParakeetAudioConfig(),
        )
        config.validate()
        return config
