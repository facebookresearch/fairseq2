# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

from fairseq2.context import RuntimeContext
from fairseq2.data import VocabularyInfo
from fairseq2.models.llama._config import LLaMARopeScalingConfig

QWEN25_MODEL_FAMILY: Final = "qwen25"

@dataclass(kw_only=True)
class Qwen25Config:
    """Abstract config without defaults"""
    model_dim: int
    """The dimensionality of the model."""

    max_seq_len: int
    """The maximum sequence length."""

    vocab_info: VocabularyInfo
    """The vocabulary information."""

    num_layers: int
    """The number of decoder layers."""

    num_attn_heads: int
    """The number of attention heads in decoder layers."""

    num_key_value_heads: int
    """The number of key/value heads for Grouped Query Attention."""

    ffn_inner_dim: int
    """The dimensionality of inner projection layers in feed-forward networks."""

    rope_theta: float
    """The coefficient of the long-term decay of the Rotary position encoder."""

    dropout_p: float = 0.0
    """The dropout probability on outputs of Transformer layers."""

    tie_embeddings: bool


def register_qwen_configs(context: RuntimeContext) -> None:
    registry = context.get_config_registry(Qwen25Config)

    arch = registry.decorator

    @arch("qwen25_7b_instruct")
    def qwen25_7b_instruct():
        vocab_info = VocabularyInfo(
            size=152064, unk_idx=None, bos_idx=151643, eos_idx=151645, pad_idx=None
        )

        config = Qwen25Config(
            model_dim=3584,
            max_seq_len=32768,
            num_layers=28,
            num_attn_heads=28,
            num_key_value_heads=4,
            ffn_inner_dim=18944,
            rope_theta=1000000.0,
            tie_embeddings=False,
            vocab_info=vocab_info
        )

        return config
    
    @arch("qwen25_7b")
    def qwen25_7b():
        config = qwen25_7b_instruct()
        config.vocab_info.eos_idx = 151643
        config.max_seq_len = 131072
        return config

    @arch("qwen25_1_5b_instruct")
    def qwen25_1_5b_instruct():
        vocab_info = VocabularyInfo(
            size=151936, unk_idx=None, bos_idx=151643, eos_idx=151645, pad_idx=None
        )

        config = Qwen25Config(
            model_dim=1536,
            max_seq_len=32768,
            num_layers=28,
            num_attn_heads=12,
            num_key_value_heads=2,
            ffn_inner_dim=8960,
            rope_theta=1000000.0,
            tie_embeddings=True,
            vocab_info=vocab_info
        )
    
        return config
    
    @arch("qwen25_1_5b")
    def qwen25_1_5b():
        config = qwen25_1_5b_instruct()
        config.vocab_info.eos_idx = 151643
        config.max_seq_len = 131072
        return config
    
    @arch("qwen25_3b_instruct")
    def qwen25_3b_instruct():
        vocab_info = VocabularyInfo(
            size=151936, unk_idx=None, bos_idx=151643, eos_idx=151645, pad_idx=None
        )
        config = Qwen25Config(
            model_dim=2048,
            max_seq_len=32768,
            num_layers=36,
            num_attn_heads=16,
            num_key_value_heads=2,
            ffn_inner_dim=11008,
            rope_theta=1000000.0,
            tie_embeddings=True,
            vocab_info=vocab_info
        )
    
        return config
    
    @arch("qwen25_3b")
    def qwen25_3b():
        config = qwen25_3b_instruct()
        config.vocab_info.eos_idx = 151643
        return config