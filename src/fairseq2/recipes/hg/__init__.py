# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Dict, List

from fairseq2.recipes.cli import Cli, RecipeCommandHandler


def check_libraries(libraries: List[str]) -> Dict[str, bool]:
    """Check if the given libraries are available."""
    availability = {}
    for lib in libraries:
        try:
            __import__(lib)
            availability[lib] = True
        except ImportError:
            availability[lib] = False
    return availability


def _setup_hg_cli(cli: Cli) -> None:
    required_libraries = ["transformers", "datasets", "evaluate", "hydra"]
    if not all(check_libraries(required_libraries).values()):
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
