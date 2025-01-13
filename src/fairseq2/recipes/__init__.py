# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
import sys
from signal import SIG_DFL, SIGINT, raise_signal, signal
from typing import Iterator

import torch
from importlib_metadata import entry_points
from torch.cuda import OutOfMemoryError

from fairseq2.error import ContractError, InternalError
from fairseq2.extensions import ExtensionError, run_extensions
from fairseq2.logging import log
from fairseq2.recipes.cli import Cli

# isort: split

from fairseq2.recipes.assets import _setup_asset_cli
from fairseq2.recipes.hg import _setup_hg_cli
from fairseq2.recipes.llama import _setup_llama_cli
from fairseq2.recipes.lm import _setup_lm_cli
from fairseq2.recipes.logging import setup_basic_logging
from fairseq2.recipes.mt import _setup_mt_cli
from fairseq2.recipes.wav2vec2 import _setup_wav2vec2_cli
from fairseq2.recipes.wav2vec2.asr import _setup_wav2vec2_asr_cli
from fairseq2.recipes.bestrq import _setup_bestrq_cli


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

    setup_basic_logging()

    setup_fairseq2()

    cli = Cli(
        name="fairseq2",
        origin_module="fairseq2",
        version=__version__,
        description="command line interface of fairseq2",
    )

    _setup_cli(cli)

    return cli.run()


def _setup_cli(cli: Cli) -> None:
    _setup_asset_cli(cli)
    _setup_lm_cli(cli)
    _setup_llama_cli(cli)
    _setup_mt_cli(cli)
    _setup_wav2vec2_cli(cli)
    _setup_wav2vec2_asr_cli(cli)
    _setup_bestrq_cli(cli)
    _setup_hg_cli(cli)

    run_extensions("fairseq2.cli", cli)
