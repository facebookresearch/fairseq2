# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.models.utils.checkpoint import convert_checkpoint, create_reverse_key_map

# isort: split

from fairseq2.models.qwen._checkpoint import QWEN_KEY_MAP
from fairseq2.models.qwen._config import QwenConfig


def export_qwen_checkpoint(
    checkpoint: dict[str, object], config: QwenConfig
) -> tuple[dict[str, object], dict[str, object]]:
    hg_config = _convert_config(config)

    hg_checkpoint = _convert_checkpoint(checkpoint, config)

    return hg_checkpoint, hg_config


def _convert_config(config: QwenConfig) -> object:
    hg_config = config.to_hg_config()

    return hg_config


def _convert_checkpoint(
    checkpoint: dict[str, object], config: QwenConfig
) -> dict[str, object]:

    checkpoint = convert_checkpoint(checkpoint, create_reverse_key_map(QWEN_KEY_MAP))

    if config.tied_embeddings:
        del checkpoint["lm_head.weight"]

    return checkpoint
