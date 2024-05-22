# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from fairseq2.recipes.cli import Cli, CliGroup, RecipeCommand
from fairseq2.recipes.lm.instruction_finetune import (
    InstructionFinetuneConfig as InstructionFinetuneConfig,
)
from fairseq2.recipes.lm.instruction_finetune import (
    instruction_finetune_presets as instruction_finetune_presets,
)
from fairseq2.recipes.lm.instruction_finetune import (
    load_instruction_finetuner as load_instruction_finetuner,
)


def _setup_lm_cli(cli: Cli) -> None:
    cli_group = CliGroup(name="lm", help="Language Model recipes")

    cli.register_group(cli_group)

    instruction_finetune_cmd = RecipeCommand(
        name="instruction-finetune",
        help="instruction-finetune a Language Model",
        loader=load_instruction_finetuner,
        preset_configs=instruction_finetune_presets,
        default_preset="llama2_7b_chat",
    )

    cli_group.register_command(instruction_finetune_cmd)
