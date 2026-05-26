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
