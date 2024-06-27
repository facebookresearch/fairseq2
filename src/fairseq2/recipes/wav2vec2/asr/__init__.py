# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from fairseq2.recipes.wav2vec2.asr.eval import (
    Wav2Vec2AsrEvalConfig as Wav2Vec2AsrEvalConfig,
)
from fairseq2.recipes.wav2vec2.asr.eval import (
    load_wav2vec2_asr_evaluator as load_wav2vec2_asr_evaluator,
)
from fairseq2.recipes.wav2vec2.asr.eval import (
    wav2vec2_asr_eval_presets as wav2vec2_asr_eval_presets,
)
from fairseq2.recipes.wav2vec2.asr.metrics import (
    Wav2Vec2AsrEvalMetricBag as Wav2Vec2AsrEvalMetricBag,
)
from fairseq2.recipes.wav2vec2.asr.metrics import (
    Wav2Vec2AsrMetricBag as Wav2Vec2AsrMetricBag,
)
from fairseq2.recipes.wav2vec2.asr.train import (
    Wav2Vec2AsrTrainConfig as Wav2Vec2AsrTrainConfig,
)
from fairseq2.recipes.wav2vec2.asr.train import (
    load_wav2vec2_asr_trainer as load_wav2vec2_asr_trainer,
)
from fairseq2.recipes.wav2vec2.asr.train import (
    wav2vec2_asr_train_presets as wav2vec2_asr_train_presets,
)
from fairseq2.recipes.wav2vec2.asr.units import (
    Wav2Vec2AsrEvalUnit as Wav2Vec2AsrEvalUnit,
)
from fairseq2.recipes.wav2vec2.asr.units import (
    Wav2Vec2AsrTrainUnit as Wav2Vec2AsrTrainUnit,
)

# isort: split

from fairseq2.recipes.cli import Cli, RecipeCommandHandler
from fairseq2.recipes.wav2vec2.asr.eval import _register_eval
from fairseq2.recipes.wav2vec2.asr.train import _register_train


def _register_wav2vec2_asr_recipes() -> None:
    _register_eval()
    _register_train()


def _setup_wav2vec2_asr_cli(cli: Cli) -> None:
    group = cli.add_group("wav2vec2_asr", help="wav2vec 2.0 ASR recipes")

    train_handler = RecipeCommandHandler(
        loader=load_wav2vec2_asr_trainer,
        preset_configs=wav2vec2_asr_train_presets,
        default_preset="base_10h",
    )

    group.add_command(
        name="train",
        handler=train_handler,
        help="train a wav2vec 2.0 ASR model",
    )

    eval_handler = RecipeCommandHandler(
        loader=load_wav2vec2_asr_evaluator,
        preset_configs=wav2vec2_asr_eval_presets,
        default_preset="base_10h",
    )

    group.add_command(
        name="eval",
        handler=eval_handler,
        help="evaluate a wav2vec 2.0 ASR model",
    )
