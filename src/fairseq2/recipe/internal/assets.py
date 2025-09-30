# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Iterator
from typing import final

from typing_extensions import override

from fairseq2.assets import (
    AssetMetadataProvider,
    AssetMetadataSource,
    AssetSourceNotFoundError,
    FileAssetMetadataLoader,
)
from fairseq2.checkpoint import ModelMetadataLoader
from fairseq2.error import raise_operational_system_error
from fairseq2.logging import log
from fairseq2.recipe.config import CommonSection


@final
class _ExtraAssetMetadataSource(AssetMetadataSource):
    def __init__(
        self, section: CommonSection, metadata_loader: FileAssetMetadataLoader
    ) -> None:
        self._section = section
        self._metadata_loader = metadata_loader

    @override
    def load(self) -> Iterator[AssetMetadataProvider]:
        for extra_path in self._section.assets.extra_paths:
            try:
                yield self._metadata_loader.load(extra_path)
            except AssetSourceNotFoundError:
                log.warning("{} pointed to by `common.assets.extra_paths` is not found.", extra_path)  # fmt: skip
            except OSError as ex:
                raise_operational_system_error(ex)


@final
class _ExtraModelMetadataSource(AssetMetadataSource):
    def __init__(
        self, section: CommonSection, metadata_loader: ModelMetadataLoader
    ) -> None:
        self._section = section
        self._metadata_loader = metadata_loader

    @override
    def load(self) -> Iterator[AssetMetadataProvider]:
        prev_checkpoint_dir = self._section.assets.prev_checkpoint_dir
        if prev_checkpoint_dir is None:
            return

        try:
            yield self._metadata_loader.load(prev_checkpoint_dir)
        except AssetSourceNotFoundError:
            log.warning("Model metadata file (model.yaml) is not found under {}. Make sure that `common.assets.prev_checkpoint_dir` points to the checkpoint directory used during training.", prev_checkpoint_dir)  # fmt: skip
        except OSError as ex:
            raise_operational_system_error(ex)
