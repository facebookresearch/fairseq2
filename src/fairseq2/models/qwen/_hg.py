# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.models.utils.checkpoint import convert_checkpoint, create_reverse_key_map

try:
    import transformers.models as transformers_models  # type: ignore[import-not-found]
    from transformers import PretrainedConfig
except ImportError:
    raise ImportError(
        "transformers package is required to fetch Qwen Config for export purpose, run `pip install transformers`"
    )

# isort: split

from fairseq2.models.qwen import QWEN_KEY_MAP
from fairseq2.models.qwen._config import QwenConfig


def export_qwen_checkpoint(
    checkpoint: dict[str, object], config: QwenConfig
) -> tuple[dict[str, object], PretrainedConfig]:
    hg_config = _convert_config(config)

    hg_checkpoint = _convert_checkpoint(checkpoint, config)

    return hg_checkpoint, hg_config


def _convert_config(config: QwenConfig) -> PretrainedConfig:
    
    config_cls = getattr(transformers_models, config.hg_config_class)

    config_mapped_to_hg = {
        "hidden_size": config.model_dim,
        "max_position_embeddings": config.max_seq_len,
        "vocab_size": config.vocab_size,
        "tie_word_embeddings": config.tied_embeddings,
        "num_hidden_layers": config.num_layers,
        "num_attention_heads": config.num_attn_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "intermediate_size": config.ffn_inner_dim,
        "head_dim": config.head_dim,
        "rope_theta": config.rope_theta,
    }

    hg_config = config_cls()

    for k, v in config_mapped_to_hg.items():
        if getattr(hg_config, k, None) is not None:
            setattr(hg_config, k, v)

    # always add architectures in the end since its used by vllm
    setattr(hg_config, "architectures", config.hg_architectures)

    return hg_config


def _convert_checkpoint(
    checkpoint: dict[str, object], config: QwenConfig
) -> dict[str, object]:

    checkpoint = convert_checkpoint(checkpoint, create_reverse_key_map(QWEN_KEY_MAP))

    if config.tied_embeddings:
        del checkpoint["lm_head.weight"]

    return checkpoint
