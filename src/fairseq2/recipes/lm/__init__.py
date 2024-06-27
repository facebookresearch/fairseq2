# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from fairseq2.recipes.lm.instruction_finetune import (
    InstructionTrainConfig as InstructionTrainConfig,
)
from fairseq2.recipes.lm.instruction_finetune import (
    instruction_train_presets as instruction_train_presets,
)
from fairseq2.recipes.lm.instruction_finetune import (
    load_instruction_trainer as load_instruction_trainer,
)
from fairseq2.recipes.lm.units import InstructionTrainUnit as InstructionTrainUnit

# isort: split

from fairseq2.recipes.cli import Cli, RecipeCommandHandler
from fairseq2.recipes.lm.chatbot import ChatbotCommandHandler
from fairseq2.recipes.lm.instruction_finetune import _register_instruction_train


def _register_lm_recipes() -> None:
    _register_instruction_train()


def _setup_lm_cli(cli: Cli) -> None:
    group = cli.add_group("lm", help="language model recipes")

    group.add_command(
        name="chatbot",
        handler=ChatbotCommandHandler(),
        help="run a terminal-based chatbot demo",
    )

    instruction_train_handler = RecipeCommandHandler(
        loader=load_instruction_trainer,
        preset_configs=instruction_train_presets,
        default_preset="llama3_8b_instruct",
    )

    group.add_command(
        name="instruction_finetune",
        handler=instruction_train_handler,
        help="train an instruction model",
    )
