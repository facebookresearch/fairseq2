# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from importlib_metadata import entry_points

from fairseq2.recipes.cli import Cli

# isort: split

from fairseq2.recipes.assets import _setup_asset_cli
from fairseq2.recipes.llama import _setup_llama_cli
from fairseq2.recipes.lm import _setup_lm_cli
from fairseq2.recipes.wav2vec2 import _setup_wav2vec2_cli
from fairseq2.recipes.wav2vec2.asr import _setup_wav2vec2_asr_cli


def main() -> None:
    """Run the command line fairseq2 program."""
    from fairseq2 import __version__, setup_extensions

    setup_extensions()

    cli = Cli(
        name="fairseq2",
        origin_module="fairseq2",
        version=__version__,
        description="command line interface of fairseq2",
    )

    _setup_cli(cli)

    cli()


def _setup_cli(cli: Cli) -> None:
    _setup_asset_cli(cli)
    _setup_lm_cli(cli)
    _setup_llama_cli(cli)
    _setup_wav2vec2_cli(cli)
    _setup_wav2vec2_asr_cli(cli)

    # Set up 3rd party CLI extensions.
    for entry_point in entry_points(group="fairseq2.cli"):
        setup_cli_extension = entry_point.load()

        try:
            setup_cli_extension(cli)
        except TypeError:
            raise RuntimeError(
                f"The entry point '{entry_point.value}' is not a valid fairseq2 CLI setup function."
            )
