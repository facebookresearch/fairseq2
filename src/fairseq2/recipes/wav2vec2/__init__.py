# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from fairseq2.recipes.cli import Cli, RecipeCommandHandler
from fairseq2.recipes.wav2vec2.train import (
    load_wav2vec2_trainer,
    wav2vec2_train_presets,
)


def _setup_wav2vec2_cli(cli: Cli) -> None:
    group = cli.add_group("wav2vec2", help="wav2vec 2.0 pretraining recipes")

    # Train
    train_handler = RecipeCommandHandler(
        loader=load_wav2vec2_trainer,
        preset_configs=wav2vec2_train_presets,
        default_preset="base_960h",
    )

    group.add_command(
        name="train",
        handler=train_handler,
        help="train a wav2vec 2.0 model",
    )
