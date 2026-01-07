# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Literal

from fairseq2.models.llama import LLaMAConfig
from fairseq2.runtime.config_registry import ConfigRegistrar
from fairseq2.runtime.dependency import DependencyContainer

OLMO_FAMILY: Final = "olmo"


@dataclass
class YaRNScaleConfig:
    """YaRN (Yet another RoPE extensioN) scaling configuration for long-context models.
    
    YaRN is applied to extend the context length of OLMO3 models from 8K to 65K.
    In OLMO3, YaRN scaling is selectively applied only to full-attention layers,
    while sliding window attention layers use regular RoPE.
    
    Reference: https://arxiv.org/abs/2309.00071
    """

    scale_factor: float = 8.0
    """The ratio between extended and original max sequence length (65536/8192 = 8.0)."""

    original_max_seq_len: int = 8192
    """Original sequence length before YaRN extension."""

    mscale: float = 1.0
    """Multiplier for attention scaling to maintain training stability."""

    mscale_all_dim: float = 0.0
    """Dimension-wise scaling parameter for YaRN."""

@dataclass(kw_only=True)
class OLMOConfig(LLaMAConfig):
    """Holds the configuration of an OLMO model (OLMO2 and OLMO3).

    This configuration supports both OLMO2 and OLMO3 architectures.
    The default values correspond to the allenai/OLMo-2-0425-1B model base architecture.
    
    OLMO2: Standard causal attention, 4K context
    OLMO3: Hybrid sliding window + full attention, 8K-65K context
    
    References:
    - OLMO2: https://arxiv.org/abs/2501.00656
    - HuggingFace: https://huggingface.co/allenai/OLMo-2-0425-1B
    """

    model_dim: int = 2048
    """The dimensionality of the model."""

    max_seq_len: int = 4096
    """The maximum sequence length."""

    vocab_size: int = 100_352
    """The size of the vocabulary."""

    pad_idx: int = 100_277
    """The index of the PAD token in the vocabulary."""

    bos_token_id: int | None = None
    """The index of the BOS token in the vocabulary."""

    eos_token_id: int = 100_257
    """The index of the EOS token in the vocabulary."""

    tied_embeddings: bool = False
    """If ``True``, ties the embedding table and the output projection layer."""

    num_layers: int = 16
    """The number of decoder layers."""

    num_attn_heads: int = 16
    """The number of attention heads in decoder layers."""

    num_key_value_heads: int = 16
    """The number of key/value heads for Grouped Query Attention.
    Olmo model use MHA, but 32B variabt use GQA
    If num_key_value_heads == num_attn_heads, the model will use Multi Head Attention (MHA),
    If num_key_value_heads == 1, the model will use Multi Query Attention (MQA),
    otherwise GQA is used.
    """

    ffn_inner_dim: int = 8192
    """The dimensionality of inner projection layers in feed-forward networks."""

    ffn_inner_dim_scale: float = 1.0
    """
    The scale factor for the dimensionality of inner projection layers in
    feed-forward networks.

    OLMO2 uses a scale of 1.0 (no scaling) unlike LLaMA which uses 2/3.
    """

    ffn_inner_dim_multiplier: float = 1.0
    """
    The multiplier for the dimensionality of inner projection layers in
    feed-forward networks.
    """

    ffn_inner_dim_multiple_of: int = 256
    """The dimensionality of inner projection layers in feed-forward networks is
    rounded up to the nearest multiple of this value."""

    rms_norm_eps: float = 1e-6
    """The epsilon value for RMSNorm layers."""

    rope_theta: float = 500_000.0
    """The coefficient of the long-term decay of the Rotary position encoder."""

    use_scaled_rope: bool = False
    """If ``True``, scales Rotary encoder frequencies to the resolver length."""

    dropout_p: float = 0.0
    """The dropout probability on outputs of Transformer layers."""


    #TODO check the init_std == initializer_range?
    init_std: float | None = None
    """
    If not ``None``, the standard deviation to initialize input embeddings and
    projection weights; otherwise, ``model_dim ** -0.5`` will be used instead.
    """
    # initializer_range: float = 0.02

    init_std_scale: Literal["none", "layer", "stack"] = "layer"
    """
    The method to use to scale ``init_std`` per layer. If 'none', no scaling
    will be applied. If 'layer', ``init_std`` will be scaled by the depth of
    the layer. If 'stack', ``init_std`` will be scaled by the total depth of
    the decoder.
    """

    #TODO check if it is used in olmo
    # shard_embed_dim: bool = False
    """If ``True``, shards the embedding dimension for tensor parallelism."""

    sliding_window: int | None = None
    """Sliding window size for local attention (OLMO3 only).
    
    If set, enables hybrid attention pattern where most layers use sliding
    window attention with this window size. Every 4th layer uses full global
    attention. The final layer always uses full global attention.
    
    OLMO3 uses sliding_window=4096 for efficient long-context processing.
    If None, all layers use full causal attention (OLMO2 behavior).
    """

    layer_types: list[Literal["sliding_attention", "full_attention"]] | None = None
    """Per-layer attention type configuration (OLMO3 only).
    
    Explicitly specifies whether each layer uses 'sliding_attention' or
    'full_attention'. If None and sliding_window is set, automatically
    generates the pattern: 3 sliding window layers, 1 full attention layer,
    with the final layer always using full attention.
    
    Length must match num_layers if specified.
    """

    yarn_scale_config: YaRNScaleConfig | None = None
    """YaRN scaling configuration for long-context models (OLMO3 only).
    
    Enables YaRN (Yet another RoPE extensioN) scaling to extend context
    length from 8K to 65K. In OLMO3, YaRN scaling is selectively applied
    only to full-attention layers; sliding window layers use regular RoPE.
    
    If None, uses standard RoPE without scaling (default for OLMO2/3 base models).
    """


def register_olmo_configs(container: DependencyContainer) -> None:
    """Register OLMO model configurations (OLMO2 and OLMO3)."""
    arch = ConfigRegistrar(container, OLMOConfig)

    # OLMO2 Model Configurations
    @arch("olmo2_1b")
    def olmo_2_1b() -> OLMOConfig:
        """OLMO2 1B model configuration."""
        # All parameters are already defaults in OLMOConfig
        return OLMOConfig()

    @arch("olmo2_7b")
    def olmo_2_7b() -> OLMOConfig:
        """OLMO2 7B model configuration."""
        config = OLMOConfig()

        # Override only the model size parameters that differ from 1B
        config.model_dim = 4096
        config.ffn_inner_dim = 11008

        config.num_layers = 32
        config.num_attn_heads = 32
        config.num_key_value_heads = 32

        return config

    @arch("olmo2_13b")
    def olmo_2_13b() -> OLMOConfig:
        """OLMO2 13B model configuration."""
        config = OLMOConfig()

        # Override only the model size parameters that differ from 1B
        config.model_dim = 5120
        config.ffn_inner_dim = 13824

        config.num_layers = 40
        config.num_attn_heads = 40
        config.num_key_value_heads = 40

        return config

    @arch("olmo2_32b")
    def olmo_2_32b() -> OLMOConfig:
        """OLMO2 32B model configuration with GQA.
        
        Uses Grouped Query Attention (GQA) for efficiency.
        """
        config = OLMOConfig()

        # Model dimensions for 32B
        config.model_dim = 5120
        config.ffn_inner_dim = 13824  # ~2.7x expansion

        config.num_layers = 64
        config.num_attn_heads = 40
        config.num_key_value_heads = 8  # GQA for efficiency

        return config

    # OLMO3 Model Configurations
    # OLMO3 extends OLMO2 with sliding window attention and support for
    # long-context variants using YaRN scaling

    @arch("olmo3_7b")
    def olmo3_7b() -> OLMOConfig:
        """OLMO3 7B model configuration with hybrid attention.
        
        Uses Multi-Head Attention (MHA) with a hybrid pattern:
        - 3 out of 4 layers use sliding window attention (window=4096)
        - Every 4th layer uses full global attention
        - Final layer always uses full global attention
        
        Max sequence length: 8,192 tokens
        """
        config = OLMOConfig()
        
        # Model dimensions (same as OLMO2 7B)
        config.model_dim = 4096
        config.ffn_inner_dim = 11008
        config.num_layers = 32
        config.num_attn_heads = 32
        config.num_key_value_heads = 32  # MHA
        
        # OLMO3-specific: Extended context and sliding window
        config.max_seq_len = 8192
        config.sliding_window = 4096
        # layer_types will be auto-generated based on sliding_window
        
        return config

    @arch("olmo3_7b_long")
    def olmo3_7b_long() -> OLMOConfig:
        """OLMO3 7B long-context model with YaRN scaling.
        
        Extends OLMO3 7B to support 65,536 token context using YaRN scaling.
        YaRN is applied only to full-attention layers; sliding window layers
        use standard RoPE.
        
        Max sequence length: 65,536 tokens
        """
        config = olmo3_7b()
        
        # Extended context with YaRN scaling
        config.max_seq_len = 65536
        config.yarn_scale_config = YaRNScaleConfig(
            scale_factor=8.0,  # 65536 / 8192
            original_max_seq_len=8192,
            mscale=1.0,
            mscale_all_dim=0.0,
        )
        
        return config

    @arch("olmo3_32b")
    def olmo3_32b() -> OLMOConfig:
        """OLMO3 32B model configuration with GQA.
        
        Uses Grouped Query Attention (GQA) with 8 key-value heads for efficiency.
        Features enhanced MLP with 5.4x expansion ratio (vs 4x in smaller models).
        
        Uses hybrid attention pattern:
        - 3 out of 4 layers use sliding window attention (window=4096)
        - Every 4th layer uses full global attention
        - Final layer always uses full global attention
        
        Max sequence length: 8,192 tokens
        """
        config = OLMOConfig()
        
        # Model dimensions
        config.model_dim = 5120
        config.ffn_inner_dim = 27648  # 5.4x expansion (5120 * 5.4 ≈ 27648)
        config.num_layers = 40
        config.num_attn_heads = 40
        config.num_key_value_heads = 8  # GQA for efficiency
        
        # OLMO3-specific: Extended context and sliding window
        config.max_seq_len = 8192
        config.sliding_window = 4096
        # layer_types will be auto-generated based on sliding_window
        
        return config

    @arch("olmo3_32b_long")
    def olmo3_32b_long() -> OLMOConfig:
        """OLMO3 32B long-context model with YaRN scaling.
        
        Extends OLMO3 32B to support 65,536 token context using YaRN scaling.
        YaRN is applied only to full-attention layers; sliding window layers
        use standard RoPE.
        
        Max sequence length: 65,536 tokens
        """
        config = olmo3_32b()
        
        # Extended context with YaRN scaling
        config.max_seq_len = 65536
        config.yarn_scale_config = YaRNScaleConfig(
            scale_factor=8.0,  # 65536 / 8192
            original_max_seq_len=8192,
            mscale=1.0,
            mscale_all_dim=0.0,
        )
        
        return config
