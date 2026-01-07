# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from fairseq2.gang import Gangs, maybe_get_current_gangs
from fairseq2.models.olmo.attention import OLMOMultiheadAttention
from fairseq2.models.olmo.config import OLMOConfig
from fairseq2.models.olmo.decoder_layer import OLMOTransformerLMDecoderLayer
from fairseq2.models.olmo.normalization import OLMORMSNorm
from fairseq2.models.olmo.yarn_rope import YaRNRotaryEncoder
from fairseq2.models.transformer import (
    CausalAttentionBias,
    FeedForwardNetwork,
    GLUFeedForwardNetwork,
    MultiheadAttention,
    TransformerEmbeddingFrontend,
    TransformerFrontend,
    create_default_sdpa,
)
from fairseq2.models.transformer_lm import (
    StandardTransformerLMDecoder,
    TransformerLM,
    TransformerLMDecoder,
    TransformerLMDecoderLayer,
)
from fairseq2.nn import (
    ColumnShardedLinear,
    Embedding,
    LayerNorm,
    Linear,
    PositionEncoder,
    Projection,
    ShardedEmbedding,
    StandardEmbedding,
    TiedProjection,
    VocabShardedEmbedding,
)
from fairseq2.nn.position_encoder import ReferenceRotaryEncoder


def create_olmo_model(config: OLMOConfig) -> TransformerLM:
    """Create an OLMO model instance (supports OLMO2 and OLMO3)."""
    gangs = maybe_get_current_gangs()

    return OLMOFactory(config, gangs).create_model()


class OLMOFactory:
    """Factory for creating OLMO models (OLMO2 and OLMO3).

    OLMO models are based on LLaMA architecture with the following differences:
    - OLMORMSNorm: multiply weight, then convert back to input dtype
    - Add Q/K Norm in attention layers.
    - Use OLMO Post-Norm in decoder layer: Attention/FFN -> Norm -> Add Residual
    - OLMO2: MHA for most models, GQA for 32B
    - OLMO3: Hybrid sliding window + full attention, MHA for 7B, GQA for 32B
    - Use a HuggingFace-style RoPE module with rotate_half
    """

    def __init__(self, config: OLMOConfig, gangs: Gangs | None = None) -> None:
        self._config = config
        self._gangs = gangs
        
        # Auto-generate layer_types if not specified but sliding_window is set
        if config.sliding_window is not None and config.layer_types is None:
            self._config.layer_types = self._generate_layer_types()
        
        # Create shared RoPE encoder (HuggingFace approach: one encoder for all layers)
        self._shared_rope_encoder = self.create_rope_encoder()
    
    def _generate_layer_types(self) -> list[str]:
        """Generate layer types for OLMO3 hybrid attention pattern.
        
        Pattern: 3 sliding window layers, 1 full attention layer, repeating.
        Final layer always uses full attention.
        
        Returns:
            List of layer types ('sliding_attention' or 'full_attention')
        """
        config = self._config
        layer_types = []
        
        for i in range(config.num_layers):
            # Final layer always uses full attention
            if i == config.num_layers - 1:
                layer_types.append("full_attention")
            # Every 4th layer (indices 3, 7, 11, ...) uses full attention
            elif (i + 1) % 4 == 0:
                layer_types.append("full_attention")
            # Other layers use sliding window attention
            else:
                layer_types.append("sliding_attention")
        
        return layer_types
    
    def _is_sliding_window_layer(self, layer_idx: int) -> bool:
        """Determine if a layer uses sliding window attention.
        
        Args:
            layer_idx: Index of the layer (0-based)
            
        Returns:
            True if layer uses sliding window attention, False for full attention
        """
        config = self._config
        
        # If no sliding window configured, all layers use full attention
        if config.sliding_window is None:
            return False
        
        # Use explicitly specified layer_types if available
        if config.layer_types is not None:
            return config.layer_types[layer_idx] == "sliding_attention"
        
        # This should not happen after __init__ auto-generation, but keep as fallback
        # Final layer always uses full attention
        if layer_idx == config.num_layers - 1:
            return False
        
        # Every 4th layer uses full attention
        return (layer_idx + 1) % 4 != 0

    def create_model(self) -> TransformerLM:
        config = self._config

        embed = self.create_embedding()

        decoder_frontend = self.create_decoder_frontend(embed)

        decoder = self.create_decoder()

        final_proj = self.create_final_projection(embed)

        return TransformerLM(
            config.model_dim,
            decoder_frontend,
            decoder,
            final_proj,
            config.pad_idx,
            config.max_seq_len,
        )

    def create_embedding(self) -> Embedding:
        config = self._config

        init_std = config.init_std

        def init_embed(embed: StandardEmbedding) -> None:
            embed_dim = embed.weight.shape[1]

            std = init_std or (embed_dim**-0.5)

            _init_truncated_normal(embed.weight, bias=None, std=std)

        embed = StandardEmbedding(
            config.vocab_size, config.model_dim, config.pad_idx, init_fn=init_embed
        )

        gangs = self._gangs

        if gangs is not None and gangs.tp.size > 1:
            if not config.shard_embed_dim:
                return VocabShardedEmbedding.from_embedding(embed, gangs.tp)

            return ShardedEmbedding.from_embedding(embed, gangs.tp)

        return embed

    def create_decoder_frontend(self, embed: Embedding) -> TransformerFrontend:
        config = self._config

        return TransformerEmbeddingFrontend(
            config.model_dim,
            embed,
            pos_encoder=None,
            no_scale=True,
            dropout_p=config.dropout_p,
        )

    def create_decoder(self) -> TransformerLMDecoder:
        config = self._config

        # Create shared RoPE encoder once (used by all layers, HuggingFace approach)
        rope_encoder = self._shared_rope_encoder

        layers = []

        for idx in range(config.num_layers):
            # All layers share the same RoPE encoder
            layer = self.create_decoder_layer(idx, rope_encoder)

            layers.append(layer)

        layer_norm = self.create_layer_norm()

        return StandardTransformerLMDecoder(layers, layer_norm)

    def create_rope_encoder(self) -> PositionEncoder:
        """Create shared rotary encoder for OLMO models.
        
        For OLMO2: Creates standard RoPE encoder.
        For OLMO3 with YaRN: Creates YaRN-scaled RoPE encoder.
        
        Note: HuggingFace uses ONE shared RoPE encoder for ALL layers, not per-layer encoders.
        
        Returns:
            PositionEncoder instance (shared across all decoder layers)
        """
        config = self._config
        head_dim = config.model_dim // config.num_attn_heads
        
        # OLMO3 long-context models use YaRN scaling
        if config.yarn_scale_config is not None:
            return YaRNRotaryEncoder(
                head_dim,
                config.max_seq_len,  # Extended length (e.g., 65536 for long-context)
                theta=config.rope_theta,
                yarn_config=config.yarn_scale_config,
            )
        
        # Standard RoPE for OLMO2 and OLMO3 base models (non-long-context)
        return ReferenceRotaryEncoder(
            head_dim,
            config.max_seq_len,
            theta=config.rope_theta,
        )

    def create_decoder_layer(
        self, layer_idx: int, rope_encoder: PositionEncoder
    ) -> TransformerLMDecoderLayer:
        """Create decoder layer with OLMO-specific Post-Norm architecture.

        OLMO models use a unique Post-Norm ordering:
        Attention/FFN -> Norm -> Add Residual

        This differs from standard Post-Norm which does:
        Attention/FFN -> Add Residual -> Norm
        """
        config = self._config

        self_attn = self.create_self_attention(layer_idx, rope_encoder)

        self_attn_layer_norm = self.create_layer_norm()

        ffn = self.create_ffn(layer_idx)

        ffn_layer_norm = self.create_layer_norm()

        return OLMOTransformerLMDecoderLayer(
            self_attn,
            self_attn_layer_norm,
            ffn,
            ffn_layer_norm,
            dropout_p=config.dropout_p,
        )

    def create_self_attention(
        self, layer_idx: int, rope_encoder: PositionEncoder
    ) -> MultiheadAttention:
        """Create self-attention layer with Q/K Norm and HuggingFace-style RoPE.

        Compared to LLaMA,
        1) OLMO adds Q/K Norm after Q and K projections.
        2) Uses HuggingFace-style RoPE (rotate_half) and keep dtypes of cos and sin.
        
        For OLMO3:
        3) Supports sliding window attention for specific layers based on layer_idx.
        """
        config = self._config

        # Determine attention type for this layer (OLMO3 hybrid attention)
        is_sliding_window = self._is_sliding_window_layer(layer_idx)
        
        if is_sliding_window and config.sliding_window is not None:
            # Create sliding window causal attention bias
            attn_bias = CausalAttentionBias(attn_window_len=config.sliding_window)
        else:
            # Full causal attention
            attn_bias = CausalAttentionBias()
            
        sdpa = create_default_sdpa(attn_bias)

        init_std = config.init_std
        std_scale_factor = self.get_std_scale_factor(layer_idx)

        def init_projection(proj: Linear) -> None:
            input_dim = proj.weight.shape[1]
            std = init_std or (input_dim**-0.5)
            _init_truncated_normal(proj.weight, proj.bias, std=std / std_scale_factor)

        head_dim = config.model_dim // config.num_attn_heads
        q_norm = self.create_layer_norm(config.num_attn_heads * head_dim)
        k_norm = self.create_layer_norm(config.num_key_value_heads * head_dim)

        return OLMOMultiheadAttention(
            config.model_dim,
            config.num_attn_heads,
            sdpa,
            num_key_value_heads=config.num_key_value_heads,
            qkv_proj_init_fn=init_projection,
            rope_encoder=rope_encoder,
            output_proj_init_fn=init_projection,
            bias=False,
            q_norm=q_norm,
            k_norm=k_norm,
            gangs=self._gangs,
        )

    def create_ffn(self, layer_idx: int) -> FeedForwardNetwork:
        config = self._config

        init_std = config.init_std

        std_scale_factor = self.get_std_scale_factor(layer_idx)

        def init_projection(proj: Linear) -> None:
            input_dim = proj.weight.shape[1]

            std = init_std or (input_dim**-0.5)

            _init_truncated_normal(proj.weight, proj.bias, std=std / std_scale_factor)

        ffn_inner_dim = int(config.ffn_inner_dim * config.ffn_inner_dim_multiplier)

        return GLUFeedForwardNetwork(
            config.model_dim,
            ffn_inner_dim,
            bias=False,
            inner_dim_scale=config.ffn_inner_dim_scale,
            inner_dim_to_multiple=config.ffn_inner_dim_multiple_of,
            proj_init_fn=init_projection,
            gangs=self._gangs,
        )

    def create_final_projection(self, embed: Embedding) -> Projection:
        config = self._config

        if config.tied_embeddings:
            if not isinstance(embed, StandardEmbedding):
                raise TypeError(
                    f"`embed` is expected to be of type `{StandardEmbedding}` when `config.tied_embeddings` is `True`, but is of type `{type(embed)}` instead."
                )

            return TiedProjection(embed.weight, bias=None)

        init_std = config.init_std

        def init_projection(proj: Linear) -> None:
            input_dim = proj.weight.shape[1]

            std = init_std or (input_dim**-0.5)

            _init_truncated_normal(proj.weight, proj.bias, std=std)

        final_proj = Linear(
            config.model_dim, config.vocab_size, bias=False, init_fn=init_projection
        )

        gangs = self._gangs

        if gangs is not None and gangs.tp.size > 1:
            return ColumnShardedLinear.from_linear(final_proj, gangs.tp)

        return final_proj

    def create_layer_norm(self, dim: int | None = None) -> LayerNorm:
        """Create OLMORMSNorm.
        OLMO RMS norm differs from Llama RMS norm in the order of operations:
        - Weight and hidden states are multiplied before converting back to the input dtype.
        """
        config = self._config

        if dim is None:
            dim = config.model_dim

        return OLMORMSNorm(
            dim,
            bias=False,
            eps=config.rms_norm_eps,
            elementwise_affine=True,
        )

    def get_std_scale_factor(self, layer_idx: int) -> float:
        config = self._config

        match config.init_std_scale:
            case "layer":
                n = layer_idx
            case "stack":
                n = config.num_layers
            case "none":
                return 1.0
            case _:
                raise ValueError(
                    f"`config.init_std_scale` must be 'none', 'layer', or 'stack', but is '{config.init_std_scale}' instead."
                )

        return (2 * (n + 1)) ** 0.5


def _init_truncated_normal(
    weight: Tensor, bias: Tensor | None, *, std: float = 1.0
) -> None:
    nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    if bias is not None:
        nn.init.zeros_(bias)
