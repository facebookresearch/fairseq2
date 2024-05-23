# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from fairseq2.recipes.cli import Cli, RecipeCommandHandler
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
    group = cli.add_group("lm", help="Language Model recipes")

    handler = RecipeCommandHandler(
        loader=load_instruction_finetuner,
        preset_configs=instruction_finetune_presets,
        default_preset="llama2_7b_chat",
    )

    group.add_command(
        "instruction_finetune", handler, help="instruction-finetune a Language Model"
    )
