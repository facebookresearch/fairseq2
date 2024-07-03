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
from fairseq2.recipes.lm.text_generate import load_text_generator, text_generate_presets


def _setup_lm_cli(cli: Cli) -> None:
    group = cli.add_group("lm", help="language model recipes")

    # Chatbot
    group.add_command(
        name="chatbot",
        handler=ChatbotCommandHandler(),
        help="run a terminal-based chatbot demo",
    )

    # Instruction Finetune
    instruction_finetune_handler = RecipeCommandHandler(
        loader=load_instruction_finetuner,
        preset_configs=instruction_finetune_presets,
        default_preset="llama3_8b_instruct",
    )

    group.add_command(
        name="instruction_finetune",
        handler=instruction_finetune_handler,
        help="instruction-finetune a language model",
    )

    # Text Generate
    text_generate_handler = RecipeCommandHandler(
        loader=load_text_generator,
        preset_configs=text_generate_presets,
        default_preset="llama3_8b_instruct",
    )

    group.add_command(
        name="generate",
        handler=text_generate_handler,
        help="generate text",
    )
