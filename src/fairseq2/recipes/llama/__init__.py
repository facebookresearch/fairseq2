# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from fairseq2.recipes.cli import Cli
from fairseq2.recipes.llama.convert import ConvertCheckpointCommand


def _setup_llama_cli(cli: Cli) -> None:
    group = cli.add_group("llama", help="LLaMA recipes")

    group.add_command(
        "convert",
        ConvertCheckpointCommand(),
        help="convert fairseq2 LLaMA checkpoints to reference checkpoints",
    )
