# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import final

from fairseq2.assets import (
    AssetMetadataProvider,
    AssetSourceNotFoundError,
    FileAssetMetadataLoader,
)
from fairseq2.checkpoint import ModelMetadataLoader
from fairseq2.error import raise_operational_system_error
from fairseq2.logging import log
from fairseq2.recipe.config import CommonSection


@final
class MaybeExtraAssetMetadataSource:
    def __init__(
        self, section: CommonSection, metadata_loader: FileAssetMetadataLoader
    ) -> None:
        self._section = section
        self._metadata_loader = metadata_loader

    def maybe_load(self) -> AssetMetadataProvider | None:
        extra_path = self._section.assets.extra_path
        if extra_path is None:
            return None

        try:
            return self._metadata_loader.load(extra_path)
        except AssetSourceNotFoundError:
            log.warning("{} pointed to by `common.assets.extra_path` is not found.", extra_path)  # fmt: skip
        except OSError as ex:
            raise_operational_system_error(ex)

        return None


@final
class MaybeExtraModelMetadataSource:
    def __init__(
        self, section: CommonSection, metadata_loader: ModelMetadataLoader
    ) -> None:
        self._section = section
        self._metadata_loader = metadata_loader

    def maybe_load(self) -> AssetMetadataProvider | None:
        checkpoint_dir = self._section.assets.checkpoint_dir
        if checkpoint_dir is None:
            return None

        try:
            return self._metadata_loader.load(checkpoint_dir)
        except AssetSourceNotFoundError:
            log.warning("Model metadata file (model.yaml) is not found under {}. Make sure that `common.assets.checkpoint_dir` points to the checkpoint directory used during training.", checkpoint_dir)  # fmt: skip
        except OSError as ex:
            raise_operational_system_error(ex)

        return None
