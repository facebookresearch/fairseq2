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

    vocab_info: VocabularyInfo = field(
        default_factory=lambda: VocabularyInfo(
            size=152064, unk_idx=None, bos_idx=151643, eos_idx=151645, pad_idx=None
        )
    )
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

@dataclass(kw_only=True)
class Qwen25HFRopeConfigPatch:
    rope_theta: float
    hidden_size: int
    num_attention_heads: int


def register_qwen_configs(context: RuntimeContext) -> None:
    registry = context.get_config_registry(Qwen25Config)

    arch = registry.decorator

    @arch("qwen25_7b")
    def qwen25_7b():
        config = Qwen25Config(
            model_dim=3584,
            max_seq_len=32768,
            num_layers=28,
            num_attn_heads=28,
            num_key_value_heads=4,
            ffn_inner_dim=18944,
            rope_theta=1000000.0
        )

        return config
    
    