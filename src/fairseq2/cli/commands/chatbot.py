# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from typing import final

import torch
from typing_extensions import override

from fairseq2.cli import CliCommandHandler
from fairseq2.cli.utils.argparse import parse_dtype
from fairseq2.context import RuntimeContext


@final
class ChatbotHandler(CliCommandHandler):
    @override
    def init_parser(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "-m",
            "--model",
            dest="model_name",
            metavar="MODEL_NAME",
            default="llama3_1_8b_instruct",
            help="instruct model name (default: %(default)s)",
        )

        parser.add_argument(
            "--dtype",
            type=parse_dtype,
            default=torch.bfloat16,
            help="data type of the model (default: %(default)s)",
        )

        parser.add_argument(
            "--tensor-parallel-size",
            type=int,
            default=1,
            help="tensor parallelism size (default: %(default)s)",
        )

        parser.add_argument(
            "--seed",
            type=int,
            default=2,
            help="random number generator seed for sequence generation (default: %(default)s)",
        )

        parser.add_argument(
            "--top-p",
            type=float,
            default=0.8,
            help="probability threshold for top-p sampling (default: %(default)s)",
        )

        parser.add_argument(
            "--temperature",
            type=float,
            default=0.6,
            help="sampling temperature (default: %(default)s)",
        )

        parser.add_argument(
            "--max-gen-len",
            type=int,
            default=2048,
            help="maximum sequence generation length (default: %(default)s)",
        )

        parser.add_argument(
            "--cluster",
            default="auto",
            help="cluster on which the recipe runs (default: %(default)s)",
        )

    @override
    def run(
        self, context: RuntimeContext, parser: ArgumentParser, args: Namespace
    ) -> int:
        return 0
