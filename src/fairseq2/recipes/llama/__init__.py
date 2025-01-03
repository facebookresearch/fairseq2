# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.recipes.cli import Cli
from fairseq2.recipes.llama.convert_checkpoint import ConvertCheckpointCommandHandler
from fairseq2.recipes.llama.write_hf_config import WriteHfConfigCommandHandler


def _setup_llama_cli(cli: Cli) -> None:
    group = cli.add_group("llama", help="LLaMA recipes")

    group.add_command(
        name="convert_checkpoint",
        handler=ConvertCheckpointCommandHandler(),
        help="convert fairseq2 LLaMA checkpoints to reference checkpoints",
    )

    group.add_command(
        name="write_hf_config",
        handler=WriteHfConfigCommandHandler(),
        help="write fairseq2 LLaMA config in Huggingface config format",
    )
