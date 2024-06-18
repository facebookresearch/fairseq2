# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from fairseq2.recipes.cli import Cli
from fairseq2.recipes.eval.local import *


def _setup_eval_cli(cli: Cli) -> None:
    group = cli.add_group("eval", help="Evaluate fairseq2 models")

    group.add_command(
        "run",
        LocalEvalCommand(),
        help="execute the evaluation in synchronous (local) mode",
    )
