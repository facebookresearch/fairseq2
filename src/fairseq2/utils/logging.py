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

from fairseq2.gang import is_coordinator_process


def setup_logging(
    *,
    log_file: Optional[Path] = None,
    debug: bool = False,
    utc_time: bool = False,
) -> None:
    """Set up logging for a training or eval job.

    :param log_file:
        The file to which logs will be written.
    :param debug:
        If ``True``, sets the log level to `DEBUG`; otherwise, to `INFO`.
    :param utc_time:
        If ``True``, logs dates and times in UTC.
    """
    handlers: List[Handler] = [StreamHandler()]  # Log to stderr.

    if log_file is not None and is_coordinator_process():
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
        except OSError as ex:
            raise RuntimeError(
                f"The log directory ({log_file.parent}) cannot be created. See nested exception for details."
            ) from ex

        handlers.append(FileHandler(log_file))  # Log to file on coordinator.

    fmt = "%(asctime)s %(levelname)s %(name)s - %(message)s"

    datefmt = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=DEBUG if debug else INFO, handlers=handlers, format=fmt, datefmt=datefmt
    )

    if utc_time:
        Formatter.converter = time.gmtime
