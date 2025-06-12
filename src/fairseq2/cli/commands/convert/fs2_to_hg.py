# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import final

import torch
from typing_extensions import override

from fairseq2.assets import (
    AssetCardError,
    AssetCardFieldNotFoundError,
    AssetCardNotFoundError,
)
from fairseq2.cli import CliArgumentError, CliCommandError, CliCommandHandler
from fairseq2.context import RuntimeContext
from fairseq2.device import CPU
from fairseq2.gang import fake_gangs
from fairseq2.logging import log
from fairseq2.models import (
    ModelConfigLoadError,
    ModelHandler,
    ModelLoadError,
    UnknownModelError,
    UnknownModelFamilyError,
    model_asset_card_error,
)
from fairseq2.recipes.common import register_extra_asset_paths
from fairseq2.recipes.config import AssetsSection


@final
class ConvertFairseq2ToHuggingFaceHandler(CliCommandHandler):
    @override
    def init_parser(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--extra-asset-path",
            type=Path,
            help="extra asset card path",
        )

        parser.add_argument(
            "--checkpoint-dir",
            type=Path,
            help="fairseq2 checkpoint directory",
        )

        parser.add_argument(
            "model",
            type=str,
            help="name of the fairseq2 model",
        )

        parser.add_argument(
            "save_dir",
            type=Path,
            help="save directory",
        )

    @override
    def run(
        self, context: RuntimeContext, parser: ArgumentParser, args: Namespace
    ) -> int:
        assets_section = AssetsSection(
            extra_path=args.extra_asset_path, checkpoint_dir=args.checkpoint_dir
        )

        register_extra_asset_paths(context, assets_section)

        try:
            dir_exists = context.file_system.exists(args.save_dir)
        except OSError as ex:
            raise CliCommandError(
                "The model cannot be saved. See the nested exception for details."
            ) from ex

        if dir_exists:
            raise CliArgumentError("save_dir", f"{args.save_dir} already exists.")

        name = args.model

        try:
            try:
                card = context.asset_store.retrieve_card(name)
            except AssetCardNotFoundError:
                raise UnknownModelError(name) from None
            except AssetCardError as ex:
                raise model_asset_card_error(name) from ex

            try:
                family = card.field("model_family").as_(str)
            except AssetCardFieldNotFoundError:
                raise UnknownModelError(name) from None
            except AssetCardError as ex:
                raise model_asset_card_error(name) from ex

            handlers = context.get_registry(ModelHandler)

            try:
                handler = handlers.get(family)
            except LookupError:
                raise UnknownModelFamilyError(family, name) from None

            if not handler.supports_hugging_face:
                raise CliArgumentError(
                    "model", f"{name} does not support Hugging Face conversion"
                )

            log.info("Loading '{}' fairseq2 model.", name)

            config = handler.load_config(card)

            gangs = fake_gangs(CPU)

            model = handler.load(card, gangs, torch.float32, config)

            state_dict = model.state_dict()

            log.info("Model loaded.", name)
        except (ModelLoadError, ModelConfigLoadError) as ex:
            raise CliCommandError(
                "The model cannot be loaded. See the nested exception for details."
            ) from ex

        save_dir = args.save_dir

        try:
            log.info("Saving Hugging Face model.")

            with TemporaryDirectory(dir=save_dir.parent) as tmp_dirname:
                tmp_save_dir = Path(tmp_dirname).joinpath(save_dir.name)

                with context.progress_reporter:
                    progress_task = context.progress_reporter.create_task(
                        name="save", total=None, start=False
                    )

                    with progress_task:
                        handler.save_as_hugging_face(tmp_save_dir, state_dict, config)

                context.file_system.move(tmp_save_dir, save_dir)

            log.info("Hugging Face model saved to {}!", save_dir)
        except (OSError, RuntimeError) as ex:
            raise CliCommandError(
                "The model cannot be saved. See the nested exception for details."
            ) from ex

        return 0
