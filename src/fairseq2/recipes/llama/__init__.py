# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from fairseq2.recipes.cli import Cli, CliGroup


def _setup_llama_cli(cli: Cli) -> None:
    llama_group = CliGroup(name="llama", help="LLaMA recipes")

    cli.register_group(llama_group)
