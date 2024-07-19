# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.recipes.cli import Cli
from fairseq2.recipes.llama.convert_checkpoint import ConvertCheckpointCommandHandler


def _setup_llama_cli(cli: Cli) -> None:
    group = cli.add_group("llama", help="LLaMA recipes")

    group.add_command(
        name="convert_checkpoint",
        handler=ConvertCheckpointCommandHandler(),
        help="convert fairseq2 LLaMA checkpoints to reference checkpoints",
    )
