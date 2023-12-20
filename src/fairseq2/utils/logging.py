# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import time
from logging import DEBUG, INFO, FileHandler, Formatter, Handler, StreamHandler
from pathlib import Path
from typing import List

from fairseq2.gang import Gang


def setup_logging(
    log_dir: Path, gang: Gang, debug: bool = False, utc: bool = False
) -> None:
    """Set up logging for a training or eval job.

    :param log_dir:
        The log output directory.
    :param gang:
        The gang of the job.
    :param debug:
        If ``True``, sets the log level to `DEBUG`; otherwise, to `INFO`.
    :param utc:
        If ``True``, logs dates and times in UTC.
    """
    if gang.rank == 0:
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
        except IOError as ex:
            raise RuntimeError(
                "The log output directory cannot be created. See nested exception for details."
            ) from ex

    gang.barrier()

    log_file = log_dir.joinpath(f"rank_{gang.rank}.log")

    # Each rank logs to its own file.
    handlers: List[Handler] = [FileHandler(log_file)]

    # On rank 0, we also print to stdout.
    if gang.rank == 0:
        handlers.append(StreamHandler())

    line_format = "%(asctime)s %(levelname)s %(name)s - %(message)s"

    datetime_format = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=DEBUG if debug else INFO,
        handlers=handlers,
        format=line_format,
        datefmt=datetime_format,
    )

    if utc:
        Formatter.converter = time.gmtime
