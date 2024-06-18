# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Union, final

from fairseq2.logging import get_log_writer
from fairseq2.recipes.cli import CliCommandHandler
from fairseq2.recipes.logging import setup_basic_logging
from fairseq2.recipes.utils.argparse import NOTSET, ParseJsonAction, bool_flag
from fairseq2.typing import override

log = get_log_writer(__name__)


@final
class LocalEvalCommand(CliCommandHandler):
    """Local evaluation mode"""

    @override
    def init_parser(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--model",
            type=Union[str, Path],
            help="card name of the model to be evaluated",
        )
        parser.add_argument(
            "--model-args",
            type=str,
            action=ParseJsonAction,
            help="arguments to override the model card, used in advanced cases (debugging, etc.)",
        )
        parser.add_argument(
            "--dataset",
            default=NOTSET,
            help="dataset, to be set explicitly in advanced cases (code inspection, profiling, etc.)",
        )
        parser.add_argument(
            "--task",
            help="name or path to the Python code of the benchmark used to evaluate the model",
        )
        parser.add_argument(
            "--task-args",
            type=str,
            action=ParseJsonAction,
            help="arguments to customize the benchmark",
        )
        parser.add_argument(
            "--dry-run",
            type=bool_flag,
            default=NOTSET,
            help="Show full input and would-be output without executing the command",
        )
        parser.add_argument(
            "--dry-run",
            type=bool_flag,
            default=False,
            help="Show full input and would-be output without executing the command",
        )
        parser.add_argument(
            "--max-samples",
            type=int,
            default=NOTSET,
            help="Maximum number of examples to evaluate (default all examples in the benchmark)",
        )
        parser.add_argument(
            "--output-dir",
            type=Union[str, Path],
            help="directory to store the evaluation results",
        )
        parser.add_argument(
            "--overwrite",
            type=bool_flag,
            default=False,
            help="Whether to overwrite the output directory if it exists",
        )
        pass

    @override
    def __call__(self, args: Namespace) -> None:
        setup_basic_logging()

        log.error("Not implemented yet")
