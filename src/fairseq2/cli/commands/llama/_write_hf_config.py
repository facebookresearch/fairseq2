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

from fairseq2.assets import (
    AssetCardError,
    AssetCardFieldNotFoundError,
    AssetCardNotFoundError,
)
from fairseq2.cli import CliArgumentError, CliCommandError, CliCommandHandler
from fairseq2.context import RuntimeContext
from fairseq2.error import InternalError
from fairseq2.logging import log
from fairseq2.models import ModelConfigLoadError, ModelHandler
from fairseq2.models.llama import LLAMA_MODEL_FAMILY, LLaMAConfig
from fairseq2.models.llama.integ import convert_to_hg_llama_config
from fairseq2.utils.file import FileMode


@final
class WriteHFLLaMAConfigHandler(CliCommandHandler):
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
            raise CliArgumentError(
                "model", f"'{args.model}' is not a known LLaMA model. Use `fairseq2 assets list` to see the available models."  # fmt: skip
            ) from None
        except AssetCardError as ex:
            raise CliCommandError(
                f"The '{args.model}' asset card cannot be read. See the nested exception for details."
            ) from ex

        try:
            family = card.field("model_family").as_(str)
        except AssetCardFieldNotFoundError:
            raise CliArgumentError(
                "name", f"'{args.model}' is not a known LLaMA model. Use `fairseq2 assets list` to see the available models."  # fmt: skip
            ) from None
        except AssetCardError as ex:
            raise CliCommandError(
                f"The '{args.model}' asset card cannot be read. See the nested exception for details."
            ) from ex

        if family != LLAMA_MODEL_FAMILY:
            raise CliArgumentError(
                "model", f"'{args.model}' is not a model of LLaMA family."  # fmt: skip
            )

        model_handlers = context.get_registry(ModelHandler)

        try:
            model_handler = model_handlers.get(LLAMA_MODEL_FAMILY)
        except LookupError:
            raise InternalError(
                "The LLaMA model handler cannot be found. Please file a bug report."
            ) from None

        try:
            model_config = model_handler.load_config(card)
        except ModelConfigLoadError as ex:
            raise CliCommandError(
                f"The configuration of the '{args.model}' model cannot be loaded. See the nested exception for details."
            ) from ex

        if not isinstance(model_config, LLaMAConfig):
            raise InternalError(
                "The model configuration type is not valid. Please file a bug report."
            )

        hg_config = convert_to_hg_llama_config(model_config)

        hg_config_file = args.output_dir.joinpath("config.json")

        def config_write_error() -> CliCommandError:
            return CliCommandError(
                f"The model configuration cannot be saved to the '{hg_config_file}' file. See the nested exception for details."
            )

        try:
            fp = context.file_system.open_text(hg_config_file, mode=FileMode.WRITE)
        except OSError as ex:
            raise config_write_error() from ex

        try:
            json.dump(hg_config, fp, indent=2, sort_keys=True)
        except OSError as ex:
            raise config_write_error() from ex
        finally:
            fp.close()

        log.info("Configuration saved to {}.", hg_config_file)

        return 0
