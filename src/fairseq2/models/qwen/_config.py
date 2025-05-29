# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from fairseq2.context import RuntimeContext

QWEN_MODEL_FAMILY: Final = "qwen"


@dataclass(kw_only=True)
class QwenConfig:
    model_dim: int = 3584
    """The dimensionality of the model."""

    max_seq_len: int = 131_072
    """The maximum sequence length."""

    vocab_size: int = 152_064
    """The size of the vocabulary."""

    tie_embeddings: bool = False
    """If ``True``, ties the embedding table and the output projection layer."""

    num_layers: int = 28
    """The number of decoder layers."""

    num_attn_heads: int = 28
    """The number of attention heads in decoder layers."""

    num_key_value_heads: int = 4
    """The number of key/value heads for Grouped Query Attention."""

    ffn_inner_dim: int = 18_944
    """The dimensionality of inner projection layers in feed-forward networks."""

    rope_theta: float = 1_000_000.0
    """The coefficient of the long-term decay of the Rotary position encoder."""

    dropout_p: float = 0.0
    """The dropout probability on outputs of Transformer layers."""


def register_qwen_configs(context: RuntimeContext) -> None:
    registry = context.get_config_registry(QwenConfig)

    arch = registry.decorator

    @arch("qwen25_7b")
    def qwen25_7b() -> QwenConfig:
        return QwenConfig()

    @arch("qwen25_7b_instruct")
    def qwen25_7b_instruct() -> QwenConfig:
        config = qwen25_7b()

        config.max_seq_len = 32_768

        return config

    @arch("qwen25_1_5b")
    def qwen25_1_5b() -> QwenConfig:
        config = qwen25_7b()

        config.model_dim = 1536
        config.vocab_size = 151_936
        config.num_attn_heads = 12
        config.num_key_value_heads = 2
        config.ffn_inner_dim = 8960
        config.tie_embeddings = True

        return config

    @arch("qwen25_1_5b_instruct")
    def qwen25_1_5b_instruct() -> QwenConfig:
        config = qwen25_1_5b()

        config.max_seq_len = 32_768

        return config

    @arch("qwen25_3b")
    def qwen25_3b() -> QwenConfig:
        config = qwen25_7b()

        config.model_dim = 2048
        config.vocab_size = 151_936
        config.num_layers = 36
        config.num_attn_heads = 16
        config.num_key_value_heads = 2
        config.ffn_inner_dim = 11_008
        config.tie_embeddings = True

        return config

    @arch("qwen25_3b_instruct")
    def qwen25_3b_instruct() -> QwenConfig:
        return qwen25_3b()
