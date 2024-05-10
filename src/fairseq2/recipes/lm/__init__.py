# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from fairseq2.recipes.cli import Cli, CliGroup, RecipeCommand


def _setup_lm_cli(cli: Cli) -> None:
    lm_group = CliGroup(name="lm", help="Language Model recipes.")

    cli.register_group(lm_group)
