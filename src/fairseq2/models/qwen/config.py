# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from fairseq2.runtime.config_registry import ConfigRegistrar
from fairseq2.runtime.dependency import DependencyContainer

QWEN_FAMILY: Final = "qwen"


@dataclass(kw_only=True)
class QwenConfig:
    model_dim: int = 3584
    """The dimensionality of the model."""

    max_seq_len: int = 32_768
    """The maximum sequence length."""

    vocab_size: int = 152_064
    """The size of the vocabulary."""

    tied_embeddings: bool = False
    """If ``True``, ties the embedding table and the output projection layer."""

    num_layers: int = 28
    """The number of decoder layers."""

    num_attn_heads: int = 28
    """The number of attention heads in decoder layers."""

    num_key_value_heads: int = 4
    """The number of key/value heads for Grouped Query Attention."""

    head_dim: int | None = None
    """
    The dimensionality of attention heads. If ``None``, uses the standard
    formula ``model_dim // num_attn_heads``.
    """

    qkv_proj_bias: bool = True
    """If ``True``, query, key, and value projections learn an additive bias."""

    q_norm: bool = False
    """If ``True``, applies Layer Normalization to projected attention queries."""

    k_norm: bool = False
    """If ``True``, applies Layer Normalization to projected attention keys."""

    ffn_inner_dim: int = 18_944
    """The dimensionality of inner projection layers in feed-forward networks."""

    rope_theta: float = 1_000_000.0
    """The coefficient of the long-term decay of the Rotary position encoder."""

    dropout_p: float = 0.0
    """The dropout probability on outputs of Transformer layers."""


def register_qwen_configs(container: DependencyContainer) -> None:
    arch = ConfigRegistrar(container, QwenConfig)

    @arch("qwen25_3b")
    def qwen25_3b() -> QwenConfig:
        config = QwenConfig()

        config.model_dim = 2048
        config.vocab_size = 151_936
        config.num_layers = 36
        config.num_attn_heads = 16
        config.num_key_value_heads = 2
        config.ffn_inner_dim = 11_008
        config.tied_embeddings = True

        return config

    @arch("qwen25_7b")
    def qwen25_7b() -> QwenConfig:
        return QwenConfig()

    @arch("qwen25_14b")
    def qwen25_14b() -> QwenConfig:
        config = QwenConfig()

        config.model_dim = 5120
        config.num_layers = 48
        config.num_attn_heads = 40
        config.num_key_value_heads = 8
        config.ffn_inner_dim = 13_824

        return config

    @arch("qwen25_32b")
    def qwen25_32b() -> QwenConfig:
        config = QwenConfig()

        config.model_dim = 5120
        config.num_layers = 64
        config.num_attn_heads = 40
        config.num_key_value_heads = 8
        config.ffn_inner_dim = 27_648

        return config

    @arch("qwen25_1_5b")
    def qwen25_1_5b() -> QwenConfig:
        config = QwenConfig()

        config.model_dim = 1536
        config.vocab_size = 151_936
        config.tied_embeddings = True
        config.num_attn_heads = 12
        config.num_key_value_heads = 2
        config.ffn_inner_dim = 8960

        return config

    @arch("qwen3_0.6b")
    def qwen3_0p6b() -> QwenConfig:
        config = QwenConfig()

        config.model_dim = 1024
        config.max_seq_len = 40_960
        config.vocab_size = 151_936
        config.tied_embeddings = True
        config.num_layers = 28
        config.num_attn_heads = 16
        config.num_key_value_heads = 8
        config.head_dim = 128
        config.qkv_proj_bias = False
        config.q_norm = True
        config.k_norm = True
        config.ffn_inner_dim = 3072
        config.rope_theta = 1_000_000

        return config

    @arch("qwen3_1.7b")
    def qwen3_1p7b() -> QwenConfig:
        config = QwenConfig()

        config.model_dim = 2048
        config.max_seq_len = 40_960
        config.vocab_size = 151_936
        config.tied_embeddings = True
        config.num_layers = 28
        config.num_attn_heads = 16
        config.num_key_value_heads = 8
        config.head_dim = 128
        config.qkv_proj_bias = False
        config.q_norm = True
        config.k_norm = True
        config.ffn_inner_dim = 6144
        config.rope_theta = 1_000_000

        return config

    @arch("qwen3_4b")
    def qwen3_4b() -> QwenConfig:
        config = QwenConfig()

        config.model_dim = 2560
        config.max_seq_len = 40_960
        config.vocab_size = 151_936
        config.tied_embeddings = True
        config.num_layers = 36
        config.num_attn_heads = 32
        config.num_key_value_heads = 8
        config.head_dim = 128
        config.qkv_proj_bias = False
        config.q_norm = True
        config.k_norm = True
        config.ffn_inner_dim = 9728
        config.rope_theta = 1_000_000

        return config

    @arch("qwen3_8b")
    def qwen3_8b() -> QwenConfig:
        config = QwenConfig()

        config.model_dim = 4096
        config.max_seq_len = 40_960
        config.vocab_size = 151_936
        config.num_layers = 36
        config.num_attn_heads = 32
        config.num_key_value_heads = 8
        config.head_dim = 128
        config.qkv_proj_bias = False
        config.q_norm = True
        config.k_norm = True
        config.ffn_inner_dim = 12_288
        config.rope_theta = 1_000_000

        return config

    @arch("qwen3_14b")
    def qwen3_14b() -> QwenConfig:
        config = QwenConfig()

        config.model_dim = 5120
        config.max_seq_len = 40_960
        config.vocab_size = 151_936
        config.num_layers = 40
        config.num_attn_heads = 40
        config.num_key_value_heads = 8
        config.head_dim = 128
        config.qkv_proj_bias = False
        config.q_norm = True
        config.k_norm = True
        config.ffn_inner_dim = 17_408
        config.rope_theta = 1_000_000

        return config

    @arch("qwen3_32b")
    def qwen3_32b() -> QwenConfig:
        config = QwenConfig()

        config.model_dim = 5120
        config.max_seq_len = 40_960
        config.vocab_size = 151_936
        config.num_layers = 64
        config.num_attn_heads = 64
        config.num_key_value_heads = 8
        config.head_dim = 128
        config.qkv_proj_bias = False
        config.q_norm = True
        config.k_norm = True
        config.ffn_inner_dim = 25_600
        config.rope_theta = 1_000_000

        return config
