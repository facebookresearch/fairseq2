# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Protocol, final, runtime_checkable

import huggingface_hub
import torch
from torch import Tensor

from fairseq2 import init_fairseq2
from fairseq2.assets import AssetCardError, AssetMetadataError, AssetStore
from fairseq2.composition import (
    ExtensionError,
    register_checkpoint_models,
    register_file_assets,
)
from fairseq2.device import CPU
from fairseq2.error import raise_operational_system_error
from fairseq2.file_system import FileSystem
from fairseq2.gang import create_fake_gangs
from fairseq2.logging import log
from fairseq2.model_checkpoint import ModelCheckpointError
from fairseq2.models import ModelFamily, ModelNotKnownError, get_model_family
from fairseq2.runtime.dependency import DependencyContainer
from fairseq2.runtime.lookup import Lookup
from fairseq2.utils.progress import ProgressReporter
from fairseq2.utils.rich import configure_rich_logging

try:
    import transformers  # type: ignore[import-not-found]
except ImportError:
    _has_transformers = False
else:
    _has_transformers = True

from fairseq2.error import OperationalError
from fairseq2.models.family import HuggingFaceExport


def save_hugging_face_model(save_dir: Path, export: HuggingFaceExport) -> None:
    if not _has_transformers:
        raise OperationalError(
            "Hugging Face Transformers is not found. Use `pip install transformers`."
        )

    from transformers import PretrainedConfig  # type: ignore[attr-defined]

    try:
        config_kls = getattr(transformers, export.config_kls_name)
    except AttributeError:
        raise TypeError(
            f"`transformers.{export.config_kls_name}` is not a type."
        ) from None

    if not issubclass(config_kls, PretrainedConfig):
        raise TypeError(
            f"`transformers.{export.config_kls_name}` is expected to be a subclass of `{PretrainedConfig}`."
        )

    config = config_kls()

    for key, value in export.config.items():
        if not hasattr(config, key):
            raise ValueError(
                f"`transformers.{export.config_kls_name}` does not have an attribute named '{key}'."
            )

        setattr(config, key, value)

    arch = export.arch

    setattr(config, "architectures", [arch] if isinstance(arch, str) else arch)

    config.save_pretrained(save_dir)

    state_dict = {}

    for key, value in export.state_dict.items():
        if not isinstance(value, Tensor):
            raise TypeError(
                f"All values in `export.state_dict` must be of type `{Tensor}`, but the value of {key} key is of type `{type(value)}` instead."
            )

        state_dict[key] = value

    huggingface_hub.save_torch_state_dict(state_dict, save_dir)


def _main() -> None:
    args = _parse_args()

    configure_rich_logging()

    try:
        _run(args)
    except ExportDirectoryAlreadyExists as ex:
        log.error("{} directory already exists.", ex.save_dir)  # fmt: skip

        sys.exit(2)
    except ModelNotKnownError as ex:
        log.error("{} is not a known model.", ex.name)  # fmt: skip

        sys.exit(2)
    except HuggingFaceNotSupportedError as ex:
        log.error("{} does not support exporting to Hugging Face.", ex.model_name)  # fmt: skip

        sys.exit(2)
    except AssetMetadataError as ex:
        log.exception("Asset metadata in {} is erroneous. See logged stack trace for details.", ex.source)  # fmt: skip

        sys.exit(1)
    except AssetCardError as ex:
        log.exception("{} asset card is erroneous. See logged stack trace for details.", ex.name)  # fmt: skip

        sys.exit(1)
    except ModelCheckpointError as ex:
        log.exception("Model checkpoint at {} is erroneous. See logged stack trace for details.", ex.path)  # fmt: skip

        sys.exit(1)
    except OperationalError:
        log.exception("Command failed due to an operational error. See logged stack trace for details.")  # fmt: skip

        sys.exit(1)
    except ExtensionError as ex:
        log.exception("{} extension failed to initialize. See logged stack trace for details.", ex.entry_point)  # fmt: skip

        sys.exit(1)
    except Exception:
        log.exception("Command failed due to an unexpected error. See logged stack trace for details and file a bug report to the corresponding author.")  # fmt: skip

        sys.exit(1)


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


def _run(args: Namespace) -> None:
    def register_command(container: DependencyContainer) -> None:
        if args.extra_asset_path:
            register_file_assets(container, args.extra_asset_path)

        if args.checkpoint_dir:
            register_checkpoint_models(container, args.checkpoint_dir)

        container.register_type(_HuggingFaceExportCommand)

        container.register_instance(_HuggingFaceSaver, save_hugging_face_model)

    resolver = init_fairseq2(extras=register_command)

    command = resolver.resolve(_HuggingFaceExportCommand)

    command.run(args.model, args.save_dir)


@runtime_checkable
class _HuggingFaceSaver(Protocol):
    def __call__(self, save_dir: Path, export: HuggingFaceExport) -> None: ...


@final
class _HuggingFaceExportCommand:
    def __init__(
        self,
        families: Lookup[ModelFamily],
        asset_store: AssetStore,
        file_system: FileSystem,
        saver: _HuggingFaceSaver,
        progress_reporter: ProgressReporter,
    ) -> None:
        self._asset_store = asset_store
        self._file_system = file_system
        self._families = families
        self._saver = saver
        self._progress_reporter = progress_reporter

    def run(self, model_name: str, save_dir: Path) -> None:
        try:
            dir_exists = self._file_system.exists(save_dir)
        except OSError as ex:
            raise_operational_system_error(ex)

        if dir_exists:
            raise ExportDirectoryAlreadyExists(save_dir)

        card = self._asset_store.maybe_retrieve_card(model_name)
        if card is None:
            raise ModelNotKnownError(model_name)

        family = get_model_family(card, self._families)

        if not family.supports_hugging_face:
            raise HuggingFaceNotSupportedError(model_name)

        gangs = create_fake_gangs(CPU)

        log.info("Loading {} model.", model_name)

        model_config = family.get_model_config(card)

        model = family.load_model(
            card, gangs, torch.float32, model_config, mmap=False, progress=True
        )

        state_dict = model.state_dict()

        log.info("Model loaded.")

        log.info("Exporting Hugging Face model.")

        export = family.convert_to_hugging_face(state_dict, model_config)

        try:
            with self._file_system.tmp_directory(save_dir.parent) as tmp_dir:
                tmp_save_dir = tmp_dir.joinpath(save_dir.name)

                with self._progress_reporter:
                    progress_task = self._progress_reporter.create_task(
                        name="export", total=None, start=False
                    )

                    with progress_task:
                        self._saver(tmp_save_dir, export)

                self._file_system.move(tmp_save_dir, save_dir)
        except OSError as ex:
            raise_operational_system_error(ex)

        log.info("Hugging Face model exported to {}!", save_dir)


class ExportDirectoryAlreadyExists(Exception):
    def __init__(self, save_dir: Path) -> None:
        super().__init__()

        self.save_dir = save_dir


class HuggingFaceNotSupportedError(Exception):
    def __init__(self, model_name: str) -> None:
        super().__init__()

        self.model_name = model_name


if __name__ == "__main__":
    _main()
