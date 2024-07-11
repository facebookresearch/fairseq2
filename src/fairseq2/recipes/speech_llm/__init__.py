# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from fairseq2.recipes.cli import Cli, RecipeCommandHandler
from fairseq2.recipes.lm.chatbot import ChatbotCommandHandler
from fairseq2.recipes.lm.instruction_finetune import (
    instruction_finetune_presets,
    load_instruction_finetuner,
)

from fairseq2.recipes.speech_llm.speech_text_align import speech_text_presets, load_speech_text_trainer

def _setup_speech_llm_cli(cli: Cli) -> None:
    group = cli.add_group("speech_llm", help="Speech LLM modality-fusion recipes")
    # Instruction Finetune
    representation_align_pretrain_handler = RecipeCommandHandler(
        loader=load_speech_text_trainer,
        preset_configs=speech_text_presets,
        default_preset="llama3_8b_speech_text_align",
    )

    group.add_command(
        name="speech_text_align",
        handler=representation_align_pretrain_handler,
        help="align speech to text repr with pre-computed alignment",
    )
