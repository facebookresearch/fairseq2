# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from fairseq2.recipes.cli import Cli, CliGroup, RecipeCommandHandler
from fairseq2.recipes.eval.configs import hf_presets


def _add_wav2vev2_asr_eval_cli(group: CliGroup) -> None:
    from fairseq2.recipes.eval.asr import load_wav2vec2_asr_evaluator

    handler = RecipeCommandHandler(
        load_wav2vec2_asr_evaluator,
        preset_configs=hf_presets,
        default_preset="librispeech_asr",
    )
    group.add_command(
        "wav2vec2-asr",
        handler,
        help="evaluate a wav2vec 2.0 ASR model in downstream benchmark",
    )


def _setup_eval_cli(cli: Cli) -> None:
    group = cli.add_group("eval", help="Evaluate fairseq2 models in downstream tasks")

    _add_wav2vev2_asr_eval_cli(group)
