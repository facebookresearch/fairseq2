# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
A command line program to convert fairseq2 models to their Hugging Face
Transformers equivalents.

.. code:: bash

    python -m fairseq2.models.utils.hg_export <fairseq2_model_name> <hg_save_dir>


See below for additional command line options.
"""

from __future__ import annotations

import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import NoReturn, Protocol, final

import torch

from fairseq2 import init_fairseq2
from fairseq2.assets import (
    AssetCardError,
    AssetDownloadError,
    AssetMetadataSourceNotFoundError,
    AssetStore,
    CorruptAssetMetadataError,
)
from fairseq2.composition import (
    ExtensionError,
    register_checkpoint_models,
    register_file_assets,
)
from fairseq2.device import CPU
from fairseq2.error import OperationalError, raise_operational_error
from fairseq2.file_system import FileSystem
from fairseq2.gang import create_fake_gangs
from fairseq2.logging import configure_logging, log
from fairseq2.model_checkpoint import CorruptModelCheckpointError
from fairseq2.models import (
    ModelFamily,
    ModelFamilyNotKnownError,
    ModelGatedError,
    _maybe_get_model_family,
)
from fairseq2.models.hg import (
    HuggingFaceConfig,
    HuggingFaceConverter,
    _LegacyHuggingFaceConverter,
    save_hugging_face_model,
)
from fairseq2.runtime.dependency import (
    DependencyContainer,
    DependencyResolver,
    activate_dependency,
    wire_object,
)
from fairseq2.runtime.lookup import Lookup
from fairseq2.utils.env import EnvironmentVariableError
from fairseq2.utils.progress import ProgressReporter


def main() -> int:
    try:
        _run()
    except CommandError as ex:
        if ex.__cause__ is None:
            log.error("{}", ex)
        else:
            log.exception("{} See logged stack trace for details.", ex)

        return 2
    except OperationalError:
        log.exception("Command failed due to an operational error. See logged stack trace for details.")  # fmt: skip

        return 1
    except Exception:
        log.exception("Command failed due to an unexpected error. See logged stack trace for details and file a bug report to the corresponding author.")  # fmt: skip

        return 1
    else:
        return 0


def _run() -> None:
    args = _parse_args()

    try:
        configure_logging(no_rich=args.no_rich)
    except EnvironmentVariableError as ex:
        raise CommandError(
            f"{ex.var_name} environment variable is not set correctly."
        ) from ex

    def extras(container: DependencyContainer) -> None:
        _register_command(container, args)

    try:
        resolver = init_fairseq2(extras=extras)
    except ExtensionError as ex:
        raise CommandError(f"{ex.entry_point} extension cannot be initialized.") from ex

    try:
        activate_dependency(resolver, AssetStore)
    except EnvironmentVariableError as ex:
        raise CommandError(
            f"{ex.var_name} environment variable is not set correctly."
        ) from ex
    except AssetMetadataSourceNotFoundError as ex:
        raise CommandError(f"{ex.source} asset source is not found.") from None
    except CorruptAssetMetadataError as ex:
        raise CommandError(f"{ex.source} asset source is erroneous.") from ex
    except AssetMetadataLoadError as ex:
        raise OperationalError("Asset store cannot be initialized.") from ex

    command = resolver.resolve(_HuggingFaceExportCommand)

    command.run(args.model, args.save_dir)


def _parse_args() -> Namespace:
    parser = ArgumentParser()

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
        "--no-rich",
        default=False,
        action="store_true",
        help="whether to disable rich text output for logging",
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

    return parser.parse_args()


def _register_command(container: DependencyContainer, args: Namespace) -> None:
    if args.extra_asset_path:
        register_file_assets(container, args.extra_asset_path)

    if args.checkpoint_dir:
        register_checkpoint_models(container, args.checkpoint_dir)

    def create_command(resolver: DependencyResolver) -> _HuggingFaceExportCommand:
        return wire_object(
            resolver, _HuggingFaceExportCommand, hg_saver=save_hugging_face_model
        )

    container.register(_HuggingFaceExportCommand, create_command)


class _HuggingFaceSaver(Protocol):
    def __call__(
        self, save_dir: Path, state_dict: dict[str, object], config: HuggingFaceConfig
    ) -> None: ...


@final
class _HuggingFaceExportCommand:
    def __init__(
        self,
        asset_store: AssetStore,
        families: Lookup[ModelFamily],
        hg_converters: Lookup[HuggingFaceConverter],
        file_system: FileSystem,
        hg_saver: _HuggingFaceSaver,
        progress_reporter: ProgressReporter,
    ) -> None:
        self._asset_store = asset_store
        self._families = families
        self._hg_converters = hg_converters
        self._file_system = file_system
        self._hg_saver = hg_saver
        self._progress_reporter = progress_reporter

    def run(self, model_name: str, save_dir: Path) -> None:
        if self._file_system.exists(save_dir):
            raise CommandError(f"{save_dir} directory already exists.")

        card = self._asset_store.maybe_retrieve_card(model_name)
        if card is None:
            raise CommandError(f"{model_name} is not a known model.")

        def raise_card_error(ex: AssetCardError) -> NoReturn:
            raise CommandError(
                f"{ex.name} asset card is erroneous. Please file a bug report to its author."
            ) from ex

        try:
            family = _maybe_get_model_family(card, self._families)
        except AssetCardError as ex:
            raise_card_error(ex)
        except ModelFamilyNotKnownError as ex:
            raise CommandError(
                f"{ex.name} family of the {model_name} model is not known."
            ) from None

        if family is None:
            raise CommandError(
                f"{card.name} asset card does not contain a model definition."
            )

        hg_converter = self._hg_converters.maybe_get(family.name)
        if hg_converter is None:
            raise CommandError(
                f"{model_name} model does not support Hugging Face export."
            )

        gangs = create_fake_gangs(CPU)

        log.info("Loading {} model.", model_name)

        try:
            config = family.get_model_config(card)
        except AssetCardError as ex:
            raise_card_error(ex)

        try:
            model = family.load_model(
                card, gangs, torch.float32, config, load_rank0_only=True, mmap=False
            )
        except OSError as ex:
            raise_operational_error(ex)
        except AssetCardError as ex:
            raise_card_error(ex)
        except AssetDownloadError as ex:
            raise_operational_error(ex)
        except CorruptModelCheckpointError as ex:
            raise CommandError(
                f"Checkpoint of the {model_name} model is erroneous."
            ) from ex
        except ModelGatedError as ex:
            raise CommandError(
                f"{model_name} model is gated. See {ex.info_url} for more information."
            ) from None

        state_dict = model.state_dict()

        log.info("Model loaded.")

        log.info("Exporting Hugging Face model.")

        if isinstance(hg_converter, _LegacyHuggingFaceConverter):
            hg_export = hg_converter._exporter(state_dict, config)

            hg_config = HuggingFaceConfig(
                hg_export.config, hg_export.config_kls_name, hg_export.arch
            )

            hg_state_dict = hg_export.state_dict
        else:
            hg_config = hg_converter.to_hg_config(config)

            hg_state_dict = hg_converter.to_hg_state_dict(state_dict, config)

        try:
            with self._file_system.tmp_directory(save_dir.parent) as tmp_dir:
                tmp_save_dir = tmp_dir.joinpath(save_dir.name)

                with self._progress_reporter:
                    progress_task = self._progress_reporter.create_task(
                        name="export", total=None, start=False
                    )

                    with progress_task:
                        self._hg_saver(tmp_save_dir, hg_state_dict, hg_config)

                self._file_system.move(tmp_save_dir, save_dir)
        except OSError as ex:
            raise_operational_error(ex)

        log.info("Hugging Face model exported to {}!", save_dir)


class CommandError(Exception):
    pass


if __name__ == "__main__":
    sys.exit(main())
