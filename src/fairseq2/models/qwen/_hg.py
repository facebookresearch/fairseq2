# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.models.utils.checkpoint import convert_checkpoint, create_reverse_key_map

# isort: split

from fairseq2.models.qwen._config import QwenConfig
from fairseq2.models.qwen._checkpoint import QWEN_KEY_MAP


def export_qwen_checkpoint(
    checkpoint: dict[str, object], config: QwenConfig
) -> tuple[dict[str, object], dict[str, object]]:
    hg_config = _convert_config(config)

    hg_checkpoint = _convert_checkpoint(checkpoint, config)

    return hg_checkpoint, hg_config


def _convert_config(config: QwenConfig) -> dict[str, object]:
    return {
        "hidden_size": config.model_dim,
        "max_position_embeddings": config.max_seq_len,
        "vocab_size": config.vocab_size,
        "tie_word_embeddings": config.tied_embeddings,
        "num_hidden_layers": config.num_layers,
        "num_attention_heads": config.num_attn_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "intermediate_size": config.ffn_inner_dim,
        "rope_theta": config.rope_theta,
    }


def _convert_checkpoint(
    checkpoint: dict[str, object], config: QwenConfig
) -> dict[str, object]:

    checkpoint = convert_checkpoint(checkpoint, create_reverse_key_map(QWEN_KEY_MAP))

    if config.tied_embeddings:
        del checkpoint["lm_head.weight"]

    return checkpoint
