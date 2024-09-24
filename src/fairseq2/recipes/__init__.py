# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from importlib_metadata import entry_points

from fairseq2.recipes.cli import Cli

# isort: split

import os

from fairseq2.dependency import DependencyResolver, get_container
from fairseq2.logging import get_log_writer
from fairseq2.recipes.assets import _setup_asset_cli
from fairseq2.recipes.hg import _setup_hg_cli
from fairseq2.recipes.llama import _setup_llama_cli
from fairseq2.recipes.lm import _setup_lm_cli
from fairseq2.recipes.logging import setup_basic_logging
from fairseq2.recipes.mt import _setup_mt_cli
from fairseq2.recipes.utils.log import exception_logger
from fairseq2.recipes.wav2vec2 import _setup_wav2vec2_cli
from fairseq2.recipes.wav2vec2.asr import _setup_wav2vec2_asr_cli

log = get_log_writer(__name__)


def main() -> None:
    """Run the command line fairseq2 program."""
    from fairseq2 import __version__, setup_fairseq2

    with exception_logger(log):
        setup_basic_logging()

        setup_fairseq2()

        cli = Cli(
            name="fairseq2",
            origin_module="fairseq2",
            version=__version__,
            description="command line interface of fairseq2",
        )

        container = get_container()

        _setup_cli(cli, container)
        _setup_cli_extensions(cli, container)

        _setup_legacy_cli_extensions(cli)

        cli(container)


def _setup_cli(cli: Cli, resolver: DependencyResolver) -> None:
    _setup_asset_cli(cli)
    _setup_lm_cli(cli)
    _setup_llama_cli(cli)
    _setup_mt_cli(cli)
    _setup_wav2vec2_cli(cli)
    _setup_wav2vec2_asr_cli(cli)
    _setup_hg_cli(cli)


def _setup_cli_extensions(cli: Cli, resolver: DependencyResolver) -> None:
    for entry_point in entry_points(group="fairseq2.extension.cli"):
        try:
            setup_cli = entry_point.load()

            setup_cli(cli, resolver)
        except TypeError:
            raise RuntimeError(
                f"The entry point '{entry_point.value}' is not a valid fairseq2 CLI extension function."
            ) from None
        except Exception as ex:
            if "FAIRSEQ2_EXTENSION_TRACE" in os.environ:
                raise RuntimeError(
                    f"The CLI extension function at '{entry_point.value}' has failed. See nested exception for details."
                ) from ex

            log.warning("The CLI extension function at '{}' has failed. Set `FAIRSEQ2_EXTENSION_TRACE` environment variable to print the stack trace.", entry_point.value)  # fmt: skip


def _setup_legacy_cli_extensions(cli: Cli) -> None:
    for entry_point in entry_points(group="fairseq2.cli"):
        try:
            setup_cli = entry_point.load()

            setup_cli(cli)
        except TypeError:
            raise RuntimeError(
                f"The entry point '{entry_point.value}' is not a valid fairseq2 CLI extension function."
            ) from None
        except Exception as ex:
            if "FAIRSEQ2_EXTENSION_TRACE" in os.environ:
                raise RuntimeError(
                    f"The CLI extension function at '{entry_point.value}' has failed. See nested exception for details."
                ) from ex

            log.warning("The CLI extension function at '{}' has failed. Set `FAIRSEQ2_EXTENSION_TRACE` environment variable to print the stack trace.", entry_point.value)  # fmt: skip
