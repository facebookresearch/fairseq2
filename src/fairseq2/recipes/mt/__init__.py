# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from fairseq2.recipes.cli import Cli, RecipeCommandHandler
from fairseq2.recipes.mt.eval import load_mt_evaluator, mt_eval_presets
from fairseq2.recipes.mt.train import load_mt_trainer, mt_train_presets
from fairseq2.recipes.mt.translate import load_text_translator, text_translate_presets
from fairseq2.recipes.utils.sweep import default_sweep_tagger


def _setup_mt_cli(cli: Cli) -> None:
    default_sweep_tagger.extend_allow_set("source_lang", "target_lang")

    group = cli.add_group("mt", help="machine translation recipes")

    # Train
    train_handler = RecipeCommandHandler(
        loader=load_mt_trainer,
        preset_configs=mt_train_presets,
        default_preset="nllb_dense_600m",
    )

    group.add_command(
        name="train",
        handler=train_handler,
        help="train a machine translation model",
    )

    # Eval
    eval_handler = RecipeCommandHandler(
        loader=load_mt_evaluator,
        preset_configs=mt_eval_presets,
        default_preset="nllb_dense_600m",
    )

    group.add_command(
        name="eval",
        handler=eval_handler,
        help="evaluate a machine translation model",
    )

    # Translate
    text_translate_handler = RecipeCommandHandler(
        loader=load_text_translator,
        preset_configs=text_translate_presets,
        default_preset="nllb_dense_600m",
    )

    group.add_command(
        name="translate",
        handler=text_translate_handler,
        help="translate text",
    )
