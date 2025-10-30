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

from transformers import AutoConfig

QWEN_OMNI_FAMILY: Final = "qwen_omni"

hf_config = AutoConfig.from_pretrained("Qwen/Qwen2.5-Omni-7B")
config_dict = hf_config.to_dict()

@dataclass(kw_only=True)
class QwenOmniConfig:
    """Temporarily refer to HF page for qwen 2.5 omni config details:
    https://huggingface.co/Qwen/Qwen2.5-Omni-7B/blob/main/config.json

    This is stored as a dict as shown and ingested
    """
    pass

def register_qwen_omni_configs(container: DependencyContainer) -> None:
    arch = ConfigRegistrar(container, QwenOmniConfig)
        
    @arch("qwen25_omni_7b")
    def qwen25_omni_7b(**config_dict) -> QwenOmniConfig:
        config = QwenOmniConfig(**config_dict)
        return config
"""
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
"""
