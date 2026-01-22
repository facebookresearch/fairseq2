# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

from fairseq2.models.llama import LLaMAConfig
from fairseq2.runtime.config_registry import ConfigRegistrar
from fairseq2.runtime.dependency import DependencyContainer

LLAMA4_FAMILY: Final = "llama4"


@dataclass(kw_only=True)
class Llama4ExpertsConfig:
    """Holds the configuration of a Llama 4 Experts"""

    num_experts: int = 16
    """The number of MoE experts."""

    use_shared_expert: bool = True
    """If ``True``, an additional single expert is used for all tokens."""

    capacity_factor: float = 1.0
    """The capacity factor of experts."""

    auto_scale: float = True
    """If ``True``, the inner dimension of experts is rescaled such that
    the number of activated params is the same as an equivalent dense layer."""

    top_k: int = 1
    """MoE sends each token to the ``top_k`` top experts."""

    interleave_moe_layer_step: int = 1
    """Llama will use a MoE layer as FFN every ``interleave_moe_layer_step``-th layer.
    If equal to 1, a MoE is used for every layer."""


@dataclass(kw_only=True)
class Llama4Config(LLaMAConfig):
    experts: Llama4ExpertsConfig = field(default_factory=lambda: Llama4ExpertsConfig())
    """If not ``None``, specifies the configuration of Mixture-of-Experts."""

    attention_chunk_size: int = 8192
    """The chunk size used for chunked attention biases."""

    use_qk_norm: bool = False
    """If ``True``, applies layer normalization to the projected query and key."""

    nope_layer_interval: int | None = None
    """
    If not ``None``, will use a NoPE layer (no positional embedding)
    instead of a RoPE layer every ``nope_layer_interval`` layers.
    """


def register_llama4_configs(container: DependencyContainer) -> None:
    arch = ConfigRegistrar(container, Llama4Config)

    @arch("llama4_scout_17b_16e")
    def llama4_scout_17b_16e() -> Llama4Config:
        config = Llama4Config()

        config.model_dim = 5120
        config.num_layers = 48
        config.num_attn_heads = 40
        config.num_key_value_heads = 8

        config.vocab_size = 202_048
        config.pad_idx = 200_018

        config.ffn_inner_dim = config.model_dim * 4
        config.ffn_inner_dim_multiplier = 1.2
        config.ffn_inner_dim_multiple_of = 2048

        config.use_qk_norm = True

        config.experts = Llama4ExpertsConfig()

        config.rope_theta = 500_000.0
        config.use_scaled_rope = True
        config.nope_layer_interval = 4
        config.rope_scale.factor = 16.0
        config.rope_scale.frequency_factors = (1.0, 1.0)

        config.shard_embed_dim = False

        return config

    @arch("llama4_maverick_17b_128e")
    def llama4_maverick_17b_128e() -> Llama4Config:
        config = llama4_scout_17b_16e()

        config.experts.num_experts = 128
        config.experts.interleave_moe_layer_step = 2

        config.use_qk_norm = False

        config.use_scaled_rope = False

        return config
