# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from fairseq2.recipes.cli import Cli, RecipeCommandHandler
from fairseq2.recipes.transformer.eval import (
    load_transformer_evaluator,
    transformer_eval_presets,
)
from fairseq2.recipes.transformer.translate import (
    load_text_translator,
    text_translate_presets,
)


def _setup_transformer_cli(cli: Cli) -> None:
    group = cli.add_group(
        "transformer", help="Transformer-based machine translation recipes"
    )

    # Eval
    eval_handler = RecipeCommandHandler(
        loader=load_transformer_evaluator,
        preset_configs=transformer_eval_presets,
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
