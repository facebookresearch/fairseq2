# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from fairseq2.config_registry import ConfigRegistry
from fairseq2.data import VocabularyInfo
from fairseq2.models.llama.factory import LLaMAConfig

llama_archs = ConfigRegistry[LLaMAConfig]()

llama_arch = llama_archs.decorator


def _7b() -> LLaMAConfig:
    return LLaMAConfig()


def _13b() -> LLaMAConfig:
    config = _7b()

    config.model_dim = 5120
    config.num_attn_heads = 40
    config.num_key_value_heads = 40
    config.ffn_inner_dim = 5120 * 4

    return config


def _33b() -> LLaMAConfig:
    config = _7b()

    config.model_dim = 6656
    config.num_layers = 60
    config.num_attn_heads = 52
    config.num_key_value_heads = 52
    config.ffn_inner_dim = 6656 * 4

    return config


def _65b() -> LLaMAConfig:
    config = _7b()

    config.model_dim = 8192
    config.num_layers = 80
    config.num_attn_heads = 64
    config.num_key_value_heads = 64
    config.ffn_inner_dim = 8192 * 4

    return config


def _llama2_7b() -> LLaMAConfig:
    config = _7b()

    config.max_seq_len = 4096

    return config


def _llama2_13b() -> LLaMAConfig:
    config = _13b()

    config.max_seq_len = 4096

    return config


def _llama2_70b() -> LLaMAConfig:
    config = _65b()

    config.max_seq_len = 4096
    config.num_key_value_heads = 8
    config.ffn_inner_dim = int(8192 * 4 * 1.3)  # See A.2.1 in LLaMA 2
    config.ffn_inner_dim_to_multiple = 4096

    return config


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


def _llama3_70b() -> LLaMAConfig:
    config = _llama2_70b()

    config.max_seq_len = 8192

    config.vocab_info = VocabularyInfo(
        size=128_256, unk_idx=None, bos_idx=128_000, eos_idx=128_001, pad_idx=None
    )

    config.rope_theta = 500_000.0

    return config


def _register_llama_archs() -> None:
    # fmt: off
    # LLaMA
    llama_archs.register("7b",  _7b)
    llama_archs.register("13b", _13b)
    llama_archs.register("33b", _33b)
    llama_archs.register("65b", _65b)

    # LLaMA 2
    llama_archs.register("llama2_7b",  _llama2_7b)
    llama_archs.register("llama2_13b", _llama2_13b)
    llama_archs.register("llama2_70b", _llama2_70b)

    # LLaMA 3
    llama_archs.register("llama3_8b",  _llama3_8b)
    llama_archs.register("llama3_70b", _llama3_70b)
    # fmt: on
