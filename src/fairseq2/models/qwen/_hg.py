# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from pathlib import Path

from fairseq2.models.utils.checkpoint import convert_checkpoint, create_reverse_key_map
from fairseq2.models.utils.hg import save_hg_checkpoint

# isort: split

from fairseq2.models.qwen._checkpoint import _QWEN_HG_KEY_MAP
from fairseq2.models.qwen._config import QwenConfig


def save_as_hg_qwen(
    save_dir: Path, checkpoint: dict[str, object], config: QwenConfig
) -> None:
    hg_checkpoint = _convert_to_hg_checkpoint(checkpoint, config)

    hg_config: dict[str, object] = {
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

    if config.hg_architecture == "Qwen3ForCausalLM":
        hg_config["head_dim"] = config.head_dim

    save_hg_checkpoint(
        save_dir,
        hg_checkpoint,
        config.hg_config_class,
        hg_config,
        config.hg_architecture,
    )


def _convert_to_hg_checkpoint(
    checkpoint: dict[str, object], config: QwenConfig
) -> dict[str, object]:
    key_map = create_reverse_key_map(_QWEN_HG_KEY_MAP)

    hg_checkpoint = convert_checkpoint(checkpoint, key_map)

    if config.tied_embeddings:
        del hg_checkpoint["lm_head.weight"]

    return hg_checkpoint
