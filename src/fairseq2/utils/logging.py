# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import time
from logging import DEBUG, INFO, FileHandler, Formatter, Handler, StreamHandler
from pathlib import Path
from typing import List, Optional

from fairseq2.gang import get_global_rank


def setup_logging(
    *,
    log_file: Optional[Path] = None,
    debug: bool = False,
    utc_time: bool = False,
) -> None:
    """Set up logging for a training or eval job.

    :param log_file:
        The file to which logs will be written. Must have a 'rank' replacement
        field; for example '/path/to/train_{rank}.log'.
    :param debug:
        If ``True``, sets the log level to `DEBUG`; otherwise, to `INFO`.
    :param utc_time:
        If ``True``, logs dates and times in UTC.
    """
    rank = get_global_rank()

    handlers: List[Handler] = [StreamHandler()]  # Log to stderr.

    if log_file is not None:
        filename = log_file.name.format(rank=rank)

        if filename == log_file.name:
            raise ValueError(
                f"`log_file` must contain a 'rank' replacement field (i.e. {{rank}}) in its filename, but is '{log_file}' instead."
            )

        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
        except OSError as ex:
            raise RuntimeError(
                f"The log directory ({log_file.parent}) cannot be created. See nested exception for details."
            ) from ex

        handler = FileHandler(log_file.with_name(filename))

        handlers.append(handler)  # Log to file.

    fmt = f"[Rank {rank}] %(asctime)s %(levelname)s %(name)s - %(message)s"

    datefmt = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=DEBUG if debug else INFO, handlers=handlers, format=fmt, datefmt=datefmt
    )

    if utc_time:
        Formatter.converter = time.gmtime
