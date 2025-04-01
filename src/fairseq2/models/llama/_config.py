# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Literal

from fairseq2.context import RuntimeContext

LLAMA_MODEL_FAMILY: Final = "llama"


@dataclass(kw_only=True)
class LLaMAConfig:
    """Holds the configuration of a LLaMA model.

    The default values correspond to the base architecture as described in
    :cite:t:`https://doi.org/10.48550/arxiv.2302.13971`.
    """

    model_dim: int = 4096
    """The dimensionality of the model."""

    max_seq_len: int = 2048
    """The maximum sequence length."""

    vocab_size: int = 32_000
    """The size of the vocabulary."""

    pad_idx: int | None = None
    """The index of the PAD symbol in the vocabulary."""

    tie_embeddings: bool = False
    """If ``True``, ties the embedding table and the output projection layer."""

    num_layers: int = 32
    """The number of decoder layers."""

    num_attn_heads: int = 32
    """The number of attention heads in decoder layers."""

    num_key_value_heads: int = 32
    """The number of key/value heads for Grouped Query Attention."""

    ffn_inner_dim: int = 4096 * 4
    """The dimensionality of inner projection layers in feed-forward networks."""

    ffn_inner_dim_scale: float = 2 / 3
    """
    The scale factor for the dimensionality of inner projection layers in
    feed-forward networks.
    """

    ffn_inner_dim_multiplier: float = 1.0
    """
    The multiplier for the dimensionality of inner projection layers in
    feed-forward networks.
    """

    ffn_inner_dim_to_multiple: int = 256
    """The dimensionality of inner projection layers in feed-forward networks is
    rounded up to the nearest multiple of this value."""

    rope_theta: float = 10_000.0
    """The coefficient of the long-term decay of the Rotary position encoder."""

    use_scaled_rope: bool = False
    """If ``True``, scales Rotary encoder frequencies to the context length."""

    rope_scaling: LLaMARopeScalingConfig = field(
        default_factory=lambda: LLaMARopeScalingConfig()
    )
    """
    If not ``None``, specifies scaling parameters for the Rotary position
    encoder, aiming to increase the context length.
    """

    init_std: float | None = None
    """
    If not ``None``, the standard deviation to initialize input embeddings and
    projection weights; otherwise, ``model_dim ** -0.5`` will be used instead.
    """

    init_std_scale: Literal["none", "layer", "stack"] = "layer"
    """
    The method to use to scale ``init_std`` per layer. If 'none', no scaling
    will be applied. If 'layer', ``init_std`` will be scaled by the depth of
    the layer. If 'stack', ``init_std`` will be scaled by the total depth of
    the decoder.
    """

    dropout_p: float = 0.0
    """The dropout probability on outputs of Transformer layers."""


@dataclass
class LLaMARopeScalingConfig:
    """
    Holds the frequency scaling configuration for the Rotary position encoder
    in LLaMA models.
    """

    factor: float = 8.0
    """
    The ratio between the intended maximum context length and the original
    maximum context length of the model.
    """

    frequency_factors: tuple[float, float] = (1.0, 4.0)
    """The factor used to define low and high frequencies."""

    original_context_length: int = 8192
    """The original context length. Defaults to LLaMA 3's context length."""


def register_llama_configs(context: RuntimeContext) -> None:
    registry = context.get_config_registry(LLaMAConfig)

    arch = registry.decorator

    @arch("7b")
    def llama1_7b() -> LLaMAConfig:
        return LLaMAConfig()

    @arch("13b")
    def llama1_13b() -> LLaMAConfig:
        config = llama1_7b()

        config.model_dim = 5120
        config.num_attn_heads = 40
        config.num_key_value_heads = 40
        config.ffn_inner_dim = 5120 * 4

        return config

    @arch("33b")
    def llama1_33b() -> LLaMAConfig:
        config = llama1_7b()

        config.model_dim = 6656
        config.num_layers = 60
        config.num_attn_heads = 52
        config.num_key_value_heads = 52
        config.ffn_inner_dim = 6656 * 4

        return config

    @arch("65b")
    def llama1_65b() -> LLaMAConfig:
        config = llama1_7b()

        config.model_dim = 8192
        config.num_layers = 80
        config.num_attn_heads = 64
        config.num_key_value_heads = 64
        config.ffn_inner_dim = 8192 * 4

        return config

    @arch("llama2_7b")
    def llama2_7b() -> LLaMAConfig:
        config = llama1_7b()

        config.max_seq_len = 4096

        return config

    @arch("llama2_13b")
    def llama2_13b() -> LLaMAConfig:
        config = llama1_13b()

        config.max_seq_len = 4096

        return config

    @arch("llama2_70b")
    def llama2_70b() -> LLaMAConfig:
        config = llama1_65b()

        config.max_seq_len = 4096
        config.num_key_value_heads = 8
        config.ffn_inner_dim = 8192 * 4
        config.ffn_inner_dim_multiplier = 1.3  # See A.2.1 in LLaMA 2
        config.ffn_inner_dim_to_multiple = 4096

        return config

    @arch("llama3_8b")
    def llama3_8b() -> LLaMAConfig:
        config = llama2_7b()

        config.max_seq_len = 8192
        config.vocab_size = 128_256
        config.pad_idx = 128_004
        config.num_key_value_heads = 8
        config.ffn_inner_dim = 4096 * 4
        config.ffn_inner_dim_multiplier = 1.3
        config.ffn_inner_dim_to_multiple = 1024
        config.rope_theta = 500_000.0

        return config

    @arch("llama3_70b")
    def llama3_70b() -> LLaMAConfig:
        config = llama2_70b()

        config.max_seq_len = 8192
        config.vocab_size = 128_256
        config.pad_idx = 128_004
        config.rope_theta = 500_000.0

        return config

    @arch("llama3_1_8b")
    def llama3_1_8b() -> LLaMAConfig:
        config = llama3_8b()

        config.max_seq_len = 131_072
        config.use_scaled_rope = True

        return config

    @arch("llama3_1_70b")
    def llama3_1_70b() -> LLaMAConfig:
        config = llama3_70b()

        config.max_seq_len = 131_072
        config.use_scaled_rope = True

        return config

    @arch("llama3_2_3b")
    def llama3_2_3b() -> LLaMAConfig:
        config = llama3_1_8b()

        config.model_dim = 3072
        config.ffn_inner_dim = 3072 * 4
        config.ffn_inner_dim_multiplier = 1.0
        config.ffn_inner_dim_to_multiple = 256
        config.num_attn_heads = 24
        config.num_key_value_heads = 8
        config.num_layers = 28
        config.use_scaled_rope = True
        config.rope_scaling.factor = 32.0

        return config

    @arch("llama3_2_1b")
    def llama3_2_1b() -> LLaMAConfig:
        config = llama3_1_8b()

        config.model_dim = 2048
        config.tie_embeddings = True
        config.ffn_inner_dim = 2048 * 4
        config.ffn_inner_dim_multiplier = 1.5
        config.ffn_inner_dim_to_multiple = 256
        config.num_attn_heads = 32
        config.num_key_value_heads = 8
        config.num_layers = 16
        config.use_scaled_rope = True
        config.rope_scaling.factor = 32.0

        return config
