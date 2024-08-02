# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from fairseq2.logging import get_log_writer
from fairseq2.recipes.cli import Cli, CliGroup, RecipeCommandHandler
from fairseq2.recipes.eval.configs import hf_presets

log = get_log_writer(__name__)


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
        help="evaluate a wav2vec 2.0 ASR model on a downstream benchmark (default: librispeech_asr)",
    )


def has_datasets() -> bool:
    try:
        import datasets  # type: ignore[attr-defined,import-untyped,import-not-found]

        return True
    except ImportError:
        log.warning(
            "`datasets` is required but not found. Please install it with `pip install datasets`. "
            "Some functions will be disabled"
        )
        return False


def has_evaluate() -> bool:
    try:
        import evaluate  # type: ignore[attr-defined,import-untyped,import-not-found]

        return True
    except ImportError:
        log.warning(
            "`evaluate` is required but not found. Please install it with `pip install evaluate`. "
            "Some functions will be disabled"
        )
        return False


def _setup_eval_cli(cli: Cli) -> None:
    group = cli.add_group("eval", help="Evaluate fairseq2 models in downstream tasks")

    if all((has_datasets(), has_evaluate())):
        _add_wav2vev2_asr_eval_cli(group)
