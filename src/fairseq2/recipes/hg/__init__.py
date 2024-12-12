# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

try:
    import datasets  # type: ignore[attr-defined,import-untyped,import-not-found]

    _has_hg_datasets = True
except ImportError:
    _has_hg_datasets = False


try:
    import evaluate  # type: ignore[attr-defined,import-untyped,import-not-found]

    _has_hg_evaluate = True
except ImportError:
    _has_hg_evaluate = False


from fairseq2.recipes.cli import Cli, RecipeCommandHandler


def _setup_hg_cli(cli: Cli) -> None:
    if not _has_hg_datasets or not _has_hg_evaluate:
        return

    group = cli.add_group("hg", help="Hugging Face recipes")

    from fairseq2.recipes.hg.asr_eval import asr_eval_presets, load_asr_evaluator

    handler = RecipeCommandHandler(
        load_asr_evaluator,
        preset_configs=asr_eval_presets,
        default_preset="default_asr",
    )

    group.add_command(
        "asr",
        handler,
        help="evaluate an ASR model (default: wav2vec2) on a downstream benchmark (default: librispeech_asr)",
    )
