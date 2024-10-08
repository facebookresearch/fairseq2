# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.recipes.cli import Cli, RecipeCommandHandler
from fairseq2.recipes.lm.chatbot import ChatbotCommandHandler
from fairseq2.recipes.lm.eval_nll import load_nll_evaluator, nll_eval_presets
from fairseq2.recipes.lm.instruction_finetune import (
    instruction_finetune_presets,
    load_instruction_finetuner,
)
from fairseq2.recipes.lm.preference_finetune import (
    load_preference_finetuner,
    preference_finetune_presets,
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
        default_preset="llama3_1_instruct",
    )

    group.add_command(
        name="instruction_finetune",
        handler=instruction_finetune_handler,
        help="instruction-finetune a language model",
    )

    # Preference Finetune
    preference_finetune_handler = RecipeCommandHandler(
        loader=load_preference_finetuner,
        preset_configs=preference_finetune_presets,
        default_preset="llama3_1_instruct",
    )

    group.add_command(
        name="preference_finetune",
        handler=preference_finetune_handler,
        help="preference-finetune a language model (e.g. DPO, SimPO).",
    )

    # Text Generate
    text_generate_handler = RecipeCommandHandler(
        loader=load_text_generator,
        preset_configs=text_generate_presets,
        default_preset="llama3_1_8b_instruct",
    )

    group.add_command(
        name="generate",
        handler=text_generate_handler,
        help="generate text",
    )

    # NLL evaluation
    nll_eval_handler = RecipeCommandHandler(
        loader=load_nll_evaluator,
        preset_configs=nll_eval_presets,
        default_preset="llama3_1_base_eval",
    )

    group.add_command(
        name="nll_eval",
        handler=nll_eval_handler,
        help="Evaluate the model and compute NLL loss over a given dataset",
    )
