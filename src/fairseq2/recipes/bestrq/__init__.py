# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.recipes.cli import Cli, RecipeCommandHandler
from fairseq2.recipes.bestrq.eval import (
    load_bestrq_evaluator,
    bestrq_eval_presets,
)
from fairseq2.recipes.bestrq.train import (
    load_bestrq_trainer,
    bestrq_train_presets,
)


def _setup_bestrq_cli(cli: Cli) -> None:
    sweep_allowed_keys = [
        "max_audio_len",
        "min_audio_len",
        "normalize_audio",
    ]

    group = cli.add_group("bestrq", help="wav2vec 2.0 pretraining recipes")

    # Train
    train_handler = RecipeCommandHandler(
        loader=load_bestrq_trainer,
        preset_configs=bestrq_train_presets,
        default_preset="base_960h",
        sweep_allowed_keys=sweep_allowed_keys,
    )

    group.add_command(
        name="train",
        handler=train_handler,
        help="train a wav2vec 2.0 model",
    )

    # Eval
    eval_handler = RecipeCommandHandler(
        loader=load_bestrq_evaluator,
        preset_configs=bestrq_eval_presets,
        default_preset="base_ls960h",
        sweep_allowed_keys=sweep_allowed_keys,
    )

    group.add_command(
        name="eval",
        handler=eval_handler,
        help="evaluate a wav2vec 2.0 model",
    )
