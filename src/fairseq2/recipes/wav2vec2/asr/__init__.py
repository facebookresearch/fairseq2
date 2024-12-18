# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.recipes.cli import Cli, RecipeCommandHandler
from fairseq2.recipes.wav2vec2.asr.eval import (
    load_wav2vec2_asr_evaluator,
    wav2vec2_asr_eval_presets,
)
from fairseq2.recipes.wav2vec2.asr.train import (
    load_wav2vec2_asr_trainer,
    wav2vec2_asr_train_presets,
)


def _setup_wav2vec2_asr_cli(cli: Cli) -> None:
    extra_sweep_keys = {
        "freeze_encoder_for_n_steps",
        "max_audio_len",
        "min_audio_len",
        "normalize_audio",
    }

    group = cli.add_group("wav2vec2_asr", help="wav2vec 2.0 ASR recipes")

    # Train
    train_handler = RecipeCommandHandler(
        loader=load_wav2vec2_asr_trainer,
        preset_configs=wav2vec2_asr_train_presets,
        default_preset="base_10h",
        extra_sweep_keys=extra_sweep_keys,
    )

    group.add_command(
        name="train",
        handler=train_handler,
        help="train a wav2vec 2.0 ASR model",
    )

    # Eval
    eval_handler = RecipeCommandHandler(
        loader=load_wav2vec2_asr_evaluator,
        preset_configs=wav2vec2_asr_eval_presets,
        default_preset="base_10h",
        extra_sweep_keys=extra_sweep_keys,
    )

    group.add_command(
        name="eval",
        handler=eval_handler,
        help="evaluate a wav2vec 2.0 ASR model",
    )
