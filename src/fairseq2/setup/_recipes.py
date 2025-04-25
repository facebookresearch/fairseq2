# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.context import RuntimeContext
from fairseq2.recipes.asr import register_asr_eval_configs
from fairseq2.recipes.lm import (
    register_instruction_finetune_configs,
    register_lm_loss_eval_configs,
    register_lm_train_configs,
    register_po_finetune_configs,
    register_text_generate_configs,
)
from fairseq2.recipes.mt import (
    register_mt_eval_configs,
    register_mt_train_configs,
    register_text_translate_configs,
)
from fairseq2.recipes.wav2vec2 import (
    register_wav2vec2_eval_configs,
    register_wav2vec2_train_configs,
)
from fairseq2.recipes.wav2vec2.asr import register_wav2vec2_asr_train_configs


def register_recipes(context: RuntimeContext) -> None:
    register_asr_eval_configs(context)
    register_instruction_finetune_configs(context)
    register_lm_loss_eval_configs(context)
    register_lm_train_configs(context)
    register_mt_eval_configs(context)
    register_mt_train_configs(context)
    register_po_finetune_configs(context)
    register_text_generate_configs(context)
    register_text_translate_configs(context)
    register_wav2vec2_asr_train_configs(context)
    register_wav2vec2_eval_configs(context)
    register_wav2vec2_train_configs(context)
