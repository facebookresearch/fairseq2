# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

from fairseq2.config_registry import ConfigRegistry
from fairseq2.context import RuntimeContext
from fairseq2.data import VocabularyInfo
from fairseq2.models.llama import LLaMAConfig, LLaMARopeScalingConfig

LLAMA4_MODEL_FAMILY: Final = "llama4"


# Not used atm
@dataclass(kw_only=True)
class LLaMA4VisionEncoderConfig:
    """Holds the configuration of a Llama 4 Vision Encoder"""
    # TODO: Add descriptions to all these
     
    image_size: int = 336
    
    patch_size: int = 14
    
    dim: int = 1408
    
    num_layers: int = 34
    
    mlp_ratio: float = 4.0
    
    output_dim: int = 4096
    
    pixel_shuffle_ratio: float = 0.5


@dataclass(kw_only=True)
class LLaMA4ExpertsConfig:
    """Holds the configuration of a Llama 4 Experts"""

    num_experts: int = 16
    """The number of MoE experts."""

    use_shared_expert: bool = True
    """If ``True``, an additional single expert is used for all tokens."""

    capacity_factor: float = 1.0
    """The capacity factor of experts."""

    auto_scale: float = True
    
    top_k: int = 1
    """MoE sends each token to the ``top_k`` top experts."""
    
    interleave_moe_layer_step: int = 1
    
    eval_with_saved_stats: bool = False
    
    expert_act_threshold: float = 0.0


@dataclass(kw_only=True)
class Llama4MetapConfig:
    """Llama 4 optional meta parameters"""

    base_width: float = 1024.0
    """The base width used in the output multiplier computation."""

    embedding_multiplier: float = 1.0
    """
    The embedding multiplier. Not implemented currently,
    since the multiplier is 1.0 in all available checkpoints.
    """


@dataclass(kw_only=True)
class LLaMA4DecoderConfig(LLaMAConfig):
    # TODO: forced to add this since FS 0.3, remove when upstreaming to 0.4
    rope_scaling: LLaMARopeScalingConfig = field(
        default_factory=lambda: LLaMARopeScalingConfig()
    )

    experts: LLaMA4ExpertsConfig = field(
        default_factory=lambda: LLaMA4ExpertsConfig()
    )
    """If not ``None``, specifies the configuration of Mixture-of-Experts."""
    
    vision_config: LLaMA4VisionEncoderConfig | None = None
    """If not ``None``, specifies the configuration of the vision encoder."""
    
    use_qk_norm: bool = False
    """If ``True``, applies layer normalization to the projected query and key."""

    metap_config: Llama4MetapConfig | None = None
    """If not ``None``, specifies the configuration of the metaparameters."""
    
    attention_chunk_size = 8192


def register_llama4_configs(context: RuntimeContext) -> None:
    registry = context.get_config_registry(LLaMA4DecoderConfig)

    arch = registry.decorator

    @arch("llama4_scout")
    def llama4_scout() -> LLaMA4DecoderConfig:
        config = LLaMA4DecoderConfig()

        config.model_dim = 5120
        config.num_layers = 48
        config.num_attn_heads = 40
        config.num_key_value_heads = 8
        
        config.vocab_size = 202_048
        # TODO: check
        config.pad_idx = 200_018
        
        config.ffn_inner_dim = config.model_dim * 4
        config.ffn_inner_dim_multiplier = 1.2
        config.ffn_inner_dim_to_multiple = 2048
        
        config.use_qk_norm = True

        config.experts = LLaMA4ExpertsConfig()

        config.rope_theta = 500_000.0
        config.use_scaled_rope = True
        config.nope_layer_interval = 4
        # TODO: check this
        config.rope_scaling.factor = 16.0
        config.rope_scaling.frequency_factors = (1.0, 1.0)
        
        # TODO: check this
        config.metap_config = Llama4MetapConfig(
            base_width=1024.0,
            embedding_multiplier=1.0,
        )
        
        return config
    
    @arch("llama4_maverick")
    def llama4_maverick() -> LLaMA4DecoderConfig:
        config = llama4_scout()
        
        config.experts.num_experts = 128
        config.experts.interleave_moe_layer_step = 2
        
        config.use_qk_norm = False

        config.use_scaled_rope = False
        config.rope_scaling = None

        return config
