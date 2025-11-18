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

OLMO2_FAMILY: Final = "olmo2"

@dataclass(kw_only=True)
class OLMO2Config(LLaMAConfig):
    """Holds the configuration of a OLMO2 model.

    The default values correspond to the allenai/OLMo-2-0425-1B model base architecture as described in
    :cite:`https://arxiv.org/abs/2501.00656`.
    :https://huggingface.co/allenai/OLMo-2-0425-1B
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

    rope_scale: bool | None = None
    # OlmoRoPEScaleConfig = field(
    #    default_factory=lambda: OlmoRoPEScaleConfig()
    #)
    """
    If not ``None``, specifies scaling parameters for the Rotary position
    encoder, aiming to increase the resolver length.
    """

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


@dataclass
class OlmoRoPEScaleConfig:
    """
    Holds the frequency scaling configuration for the Rotary position encoder
    in Olmo models.
    """

    factor: float = 8.0
    """
    The ratio between the intended maximum resolver length and the original
    maximum resolver length of the model.
    """

    frequency_factors: tuple[float, float] = (1.0, 4.0)
    """The factor used to define low and high frequencies."""

    original_context_length: int = 8192
    """The original resolver length. Defaults to LLaMA 3's resolver length."""


def register_olmo2_configs(container: DependencyContainer) -> None:
    """Register OLMO2 model configurations."""
    arch = ConfigRegistrar(container, OLMO2Config)

    @arch("olmo2-0425-1b")
    def olmo_2_0425_1b() -> OLMO2Config:
        """OLMO2 0425 1B model configuration."""
        # All parameters are already defaults in OLMO2Config
        return OLMO2Config()

    @arch("olmo2-1124-7b")
    def olmo_2_1124_7b() -> OLMO2Config:
        """OLMO2 1124 7B model configuration."""
        config = OLMO2Config()

        # Override only the model size parameters that differ from 1B
        config.model_dim = 4096
        config.ffn_inner_dim = 11008

        config.num_layers = 32
        config.num_attn_heads = 32
        config.num_key_value_heads = 32

        return config

    @arch("olmo2-1124-13b")
    def olmo_2_1124_13b() -> OLMO2Config:
        """OLMO2 1124 13B model configuration."""
        config = OLMO2Config()

        # Override only the model size parameters that differ from 1B
        config.model_dim = 5120
        config.ffn_inner_dim = 13824

        config.num_layers = 40
        config.num_attn_heads = 40
        config.num_key_value_heads = 40

        return config
