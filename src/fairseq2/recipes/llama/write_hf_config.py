# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import final

from typing_extensions import override

from fairseq2.assets import default_asset_store
from fairseq2.logging import get_log_writer
from fairseq2.models.llama import load_llama_config
from fairseq2.models.llama.factory import convert_to_huggingface_config
from fairseq2.recipes.cli import CliCommandHandler
from fairseq2.recipes.console import get_error_console
from fairseq2.setup import setup_fairseq2

log = get_log_writer(__name__)


@final
class WriteHfConfigCommandHandler(CliCommandHandler):
    """Writes fairseq2 LLaMA config files in Huggingface format."""

    @override
    def init_parser(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--model",
            metavar="ARCH_NAME",
            help="model name to fetch architecture to generate config.json",
        )

        parser.add_argument(
            "output_dir",
            type=Path,
            help="output directory to store reference checkpoint",
        )

    @override
    def run(self, args: Namespace) -> int:
        setup_fairseq2()

        arch = (
            default_asset_store.retrieve_card(args.model).field("model_arch").as_(str)
        )

        if arch:
            model_config = load_llama_config(args.model)
        else:
            model_config = None

        if model_config is None:
            log.error("Model config could not be retrieved for model {}", args.model)

            sys.exit(1)
        
        args.output_dir.mkdir(parents=True, exist_ok=True)

        # Convert and write the config
        with get_error_console().status("[bold green]Writing config...") as status:

            config = convert_to_huggingface_config(arch, model_config)
            to_write = json.dumps(config, indent=2, sort_keys=True) + "\n"

            json_file = args.output_dir.joinpath("config.json")

            try:
                with json_file.open("w") as fp:
                    fp.write(to_write)
            except OSError as ex:
                raise RuntimeError(
                    f"The file config.json cannot be saved. See the nested exception for details."
                ) from ex

            log.info("Config converted and saved in {}", json_file)

        return 0
