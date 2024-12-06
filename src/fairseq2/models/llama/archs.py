# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.data import VocabularyInfo
from fairseq2.models.llama.factory import LLaMAConfig, llama_arch


@llama_arch("7b")
def _7b() -> LLaMAConfig:
    return LLaMAConfig()


@llama_arch("13b")
def _13b() -> LLaMAConfig:
    config = _7b()

    config.model_dim = 5120
    config.num_attn_heads = 40
    config.num_key_value_heads = 40
    config.ffn_inner_dim = 5120 * 4

    return config


@llama_arch("33b")
def _33b() -> LLaMAConfig:
    config = _7b()

    config.model_dim = 6656
    config.num_layers = 60
    config.num_attn_heads = 52
    config.num_key_value_heads = 52
    config.ffn_inner_dim = 6656 * 4

    return config


@llama_arch("65b")
def _65b() -> LLaMAConfig:
    config = _7b()

    config.model_dim = 8192
    config.num_layers = 80
    config.num_attn_heads = 64
    config.num_key_value_heads = 64
    config.ffn_inner_dim = 8192 * 4

    return config


@llama_arch("llama2_7b")
def _llama2_7b() -> LLaMAConfig:
    config = _7b()

    config.max_seq_len = 4096

    return config


@llama_arch("llama2_13b")
def _llama2_13b() -> LLaMAConfig:
    config = _13b()

    config.max_seq_len = 4096

    return config


@llama_arch("llama2_70b")
def _llama2_70b() -> LLaMAConfig:
    config = _65b()

    config.max_seq_len = 4096
    config.num_key_value_heads = 8
    config.ffn_inner_dim = int(8192 * 4 * 1.3)  # See A.2.1 in LLaMA 2
    config.ffn_inner_dim_to_multiple = 4096

    return config


@llama_arch("llama3_8b")
def _llama3_8b() -> LLaMAConfig:
    config = _llama2_7b()

    config.max_seq_len = 8192

    config.vocab_info = VocabularyInfo(
        size=128_256, unk_idx=None, bos_idx=128_000, eos_idx=128_001, pad_idx=None
    )

    config.num_key_value_heads = 8
    config.ffn_inner_dim = int(4096 * 4 * 1.3)
    config.ffn_inner_dim_to_multiple = 1024
    config.rope_theta = 500_000.0

    return config


@llama_arch("llama3_70b")
def _llama3_70b() -> LLaMAConfig:
    config = _llama2_70b()

    config.max_seq_len = 8192

    config.vocab_info = VocabularyInfo(
        size=128_256, unk_idx=None, bos_idx=128_000, eos_idx=128_001, pad_idx=None
    )

    config.rope_theta = 500_000.0

    return config


@llama_arch("llama3_1_8b")
def _llama3_1_8b() -> LLaMAConfig:
    config = _llama3_8b()

    config.max_seq_len = 131_072
    config.use_scaled_rope = True

    return config


@llama_arch("llama3_1_70b")
def _llama3_1_70b() -> LLaMAConfig:
    config = _llama3_70b()

    config.max_seq_len = 131_072
    config.use_scaled_rope = True

    return config


@llama_arch("llama3_2_3b")
def _llama3_2_3b() -> LLaMAConfig:
    config = _llama3_1_8b()

    config.model_dim = 3072
    config.ffn_inner_dim = int(3072 * 4 * 1.0)
    config.ffn_inner_dim_to_multiple = 256
    config.num_attn_heads = 24
    config.num_key_value_heads = 8
    config.num_layers = 28

    return config


@llama_arch("llama3_2_1b")
def _llama3_2_1b() -> LLaMAConfig:
    config = _llama3_1_8b()

    config.model_dim = 2048
    config.ffn_inner_dim = int(2048 * 4 * 1.5)
    config.ffn_inner_dim_to_multiple = 256
    config.num_attn_heads = 32
    config.num_key_value_heads = 8
    config.num_layers = 16

    return config
