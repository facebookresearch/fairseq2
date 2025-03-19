# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
import os
import time
from logging import DEBUG, INFO, Formatter, Handler, NullHandler

from rich.logging import RichHandler

from fairseq2.cli.utils.rich import get_error_console
from fairseq2.logging import LoggingSetupError
from fairseq2.utils.env import InvalidEnvironmentVariableError, get_rank


def setup_logging(*, debug: bool = False, utc_time: bool = False) -> None:
    try:
        rank = get_rank(os.environ)
    except InvalidEnvironmentVariableError as ex:
        raise LoggingSetupError(
            "The rank of the process cannot be determined. See the nested exception for details."
        ) from ex

    level = DEBUG if debug else INFO

    if utc_time:
        Formatter.converter = time.gmtime

    handlers: list[Handler] = []

    if rank == 0:
        console = get_error_console()

        handler = RichHandler(console=console, show_path=False, keywords=[])

        handler.setFormatter(Formatter("%(name)s - %(message)s"))

        handlers.append(handler)
    else:
        handlers.append(NullHandler())

    datefmt = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(level=level, handlers=handlers, datefmt=datefmt, force=True)
