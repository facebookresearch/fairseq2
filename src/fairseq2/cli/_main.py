# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
import sys
from signal import raise_signal, SIG_DFL, SIGINT, signal

import torch

from fairseq2 import setup_fairseq2

# isort: split

from fairseq2.cli._logging import setup_logging
from fairseq2.cli._setup import setup_cli
from fairseq2.cli.utils.rich import create_rich_progress_reporter
from fairseq2.error import ContractError, InternalError
from fairseq2.extensions import ExtensionError
from fairseq2.logging import log, LoggingSetupError
from fairseq2.setup import SetupError
from fairseq2.utils.env import get_rank, InvalidEnvironmentVariableError
from torch.cuda import OutOfMemoryError


def main() -> None:
    """Runs the command line fairseq2 program."""
    exit_code = 1

    # try:
    exit_code = _run()
    # except KeyboardInterrupt:
    #     log.info("Command canceled!")

    #     signal(SIGINT, SIG_DFL)

    #     raise_signal(SIGINT)
    # except OutOfMemoryError:
    #     s = torch.cuda.memory_summary()

    #     log.exception("CUDA out of memory. See logged memory stats.\n{}", s)
    # except InternalError:
    #     log.exception("Command failed with an unexpected internal error. Please file a bug report.")  # fmt: skip
    # except ContractError:
    #     log.exception("Command failed with an unexpected internal error caused by an extension. Please file a bug report to the corresponding extension author.")  # fmt: skip
    # except Exception:
    #     log.exception("Command failed with an unexpected error. See the logged stack trace for details.")  # fmt: skip

    sys.exit(exit_code)


def _run() -> int:
    try:
        setup_logging()
    except LoggingSetupError:
        log.exception("Command setup failed. See the logged stack trace for details.")

        return 1

    try:
        try:
            rank = get_rank(os.environ)
        except InvalidEnvironmentVariableError as ex:
            raise SetupError(
                "The rank of the process cannot be determined. See the nested exception for details."
            ) from ex

        progress_reporter = create_rich_progress_reporter(rank)

        context = setup_fairseq2(progress_reporter)

        cli = setup_cli(context)
    except SetupError:
        log.exception("Command setup failed. See the logged stack trace for details.")

        return 1
    except ExtensionError as ex:
        log.exception("{} extension failed to load. See the logged stack trace for details.", ex.entry_point)  # fmt: skip

        return 1

    return cli.run(context)


if __name__ == "__main__":
    main()
