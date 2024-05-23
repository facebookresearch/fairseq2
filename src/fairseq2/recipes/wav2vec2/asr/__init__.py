# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from fairseq2.recipes.cli import Cli, CliGroup, RecipeCommand
from fairseq2.recipes.wav2vec2.asr.eval import (
    load_wav2vec2_asr_evaluator,
    wav2vec2_asr_eval_presets,
)
from fairseq2.recipes.wav2vec2.asr.train import (
    load_wav2vec2_asr_trainer,
    wav2vec2_asr_train_presets,
)


def _setup_wav2vec2_asr_cli(cli: Cli) -> None:
    cli_group = CliGroup(name="wav2vec2_asr", help="wav2vec 2.0 ASR recipes")

    cli.register_group(cli_group)

    train_cmd = RecipeCommand(
        name="train",
        help="train a wav2vec 2.0 ASR model",
        loader=load_wav2vec2_asr_trainer,
        preset_configs=wav2vec2_asr_train_presets,
        default_preset="base_10h",
    )

    cli_group.register_command(train_cmd)

    eval_cmd = RecipeCommand(
        name="eval",
        help="evaluate a wav2vec 2.0 ASR model",
        loader=load_wav2vec2_asr_evaluator,
        preset_configs=wav2vec2_asr_eval_presets,
        default_preset="base_10h",
    )

    cli_group.register_command(eval_cmd)
