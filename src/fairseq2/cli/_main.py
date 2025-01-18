# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import sys
from signal import SIG_DFL, SIGINT, raise_signal, signal

import torch
from torch.cuda import OutOfMemoryError

from fairseq2.cli._cli import Cli
from fairseq2.cli._logging import setup_logging
from fairseq2.cli.setup import setup_cli
from fairseq2.error import ContractError, InternalError
from fairseq2.extensions import ExtensionError
from fairseq2.logging import log


def main() -> None:
    """Run the command line fairseq2 program."""
    exit_code = 1

    try:
        exit_code = _run()
    except KeyboardInterrupt:
        log.info("The command has been canceled!")

        signal(SIGINT, SIG_DFL)

        raise_signal(SIGINT)
    except OutOfMemoryError:
        s = torch.cuda.memory_summary()

        log.exception("CUDA run out of memory. See the logged memory stats.\n{}", s)
    except ExtensionError as ex:
        log.exception("The '{}' extension has failed to load. See the logged stack trace for details.", ex.entry_point)  # fmt: skip
    except InternalError:
        log.exception("The command has failed with an unexpected internal error. Please file a bug report.")  # fmt: skip
    except ContractError:
        log.exception("The command has failed with an unexpected internal error caused by an extension. See the logged stack trace for details and file a bug report to the corresponding extension author.")  # fmt: skip
    except Exception:
        log.exception("The command has failed with an unexpected error. See the logged stack trace for details.")  # fmt: skip

    sys.exit(exit_code)


def _run() -> int:
    from fairseq2 import __version__, setup_fairseq2

    setup_logging()

    context = setup_fairseq2()

    cli = Cli(
        name="fairseq2",
        origin_module="fairseq2",
        version=__version__,
        description="command line interface of fairseq2",
    )

    setup_cli(cli)

    return cli.run(context)
