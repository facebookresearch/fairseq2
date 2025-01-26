# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import final

from typing_extensions import override

from fairseq2.assets import AssetCardNotFoundError
from fairseq2.cli import CliCommandHandler
from fairseq2.context import RuntimeContext
from fairseq2.logging import log
from fairseq2.models import ModelHandler, UnknownModelArchitectureError
from fairseq2.models.llama import LLAMA_MODEL_FAMILY, LLaMAConfig
from fairseq2.models.llama.integ import convert_to_hg_llama_config
from fairseq2.utils.file import FileMode


@final
class WriteHFLLaMAConfigHandler(CliCommandHandler):
    """Writes fairseq2 LLaMA configurations in Hugging Face format."""

    @override
    def init_parser(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "model",
            type=str,
            help="model for which to generate config.json",
        )

        parser.add_argument(
            "output_dir",
            type=Path,
            help="output directory",
        )

    @override
    def run(
        self, context: RuntimeContext, parser: ArgumentParser, args: Namespace
    ) -> int:
        try:
            card = context.asset_store.retrieve_card(args.model)
        except AssetCardNotFoundError:
            parser.error("unknown LLaMA model. Use `fairseq2 assets list` to see the available models.")  # fmt: skip

            return 1

        model_handlers = context.get_registry(ModelHandler)

        try:
            model_handler = model_handlers.get(LLAMA_MODEL_FAMILY)
        except LookupError:
            log.error("LLaMA model handler cannot be found. Please file a bug report.")  # fmt: skip

            return 1

        try:
            model_config = model_handler.load_config(card)
        except UnknownModelArchitectureError:
            log.error("Model has an unknown architecture. Please file a bug report to the model author.")  # fmt: skip

            return 1

        if not isinstance(model_config, LLaMAConfig):
            log.error("Model configuration has an invalid type. Please file a bug report.")  # fmt: skip

            return 1

        hg_config = convert_to_hg_llama_config(model_config)

        hg_config_file = args.output_dir.joinpath("config.json")

        try:
            fp = context.file_system.open_text(hg_config_file, mode=FileMode.WRITE)
        except OSError:
            log.exception("Configuration cannot be saved. See the logged stack trace for details.")  # fmt: skip

            return 1

        try:
            json.dump(hg_config, fp, indent=2, sort_keys=True)
        except OSError:
            log.exception("Configuration cannot be saved. See the logged stack trace for details.")  # fmt: skip

            return 1
        finally:
            fp.close()

        log.info("Configuration saved in {}.", hg_config_file)

        return 0
