# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any, TypeVar, final

from fairseq2.assets import AssetDownloadManager, StandardAssetStore
from fairseq2.config_registry import ConfigRegistry
from fairseq2.registry import Registry
from fairseq2.utils.file import FileSystem
from fairseq2.utils.progress import ProgressReporter
from fairseq2.utils.stopwatch import Stopwatch

T = TypeVar("T")


@final
class RuntimeContext:
    _env: MutableMapping[str, str]
    _asset_store: StandardAssetStore
    _asset_download_manager: AssetDownloadManager
    _file_system: FileSystem
    _progress_reporter: ProgressReporter
    _registries: dict[type, Registry[Any]]
    _config_registries: dict[type, ConfigRegistry[Any]]
    _wall_watch: Stopwatch

    def __init__(
        self,
        env: MutableMapping[str, str],
        asset_store: StandardAssetStore,
        asset_download_manager: AssetDownloadManager,
        file_system: FileSystem,
        progress_reporter: ProgressReporter,
    ) -> None:
        self._env = env
        self._asset_store = asset_store
        self._asset_download_manager = asset_download_manager
        self._file_system = file_system
        self._progress_reporter = progress_reporter
        self._registries = {}
        self._config_registries = {}

        self._wall_watch = Stopwatch()

    @property
    def env(self) -> MutableMapping[str, str]:
        return self._env

    @property
    def asset_store(self) -> StandardAssetStore:
        return self._asset_store

    @property
    def asset_download_manager(self) -> AssetDownloadManager:
        return self._asset_download_manager

    @property
    def file_system(self) -> FileSystem:
        return self._file_system

    @property
    def progress_reporter(self) -> ProgressReporter:
        return self._progress_reporter

    def get_registry(self, kls: type[T]) -> Registry[T]:
        registry = self._registries.get(kls)
        if registry is None:
            registry = Registry(kls)

            self._registries[kls] = registry

        return registry

    def get_config_registry(self, config_kls: type[T]) -> ConfigRegistry[T]:
        registry = self._config_registries.get(config_kls)
        if registry is None:
            registry = ConfigRegistry(config_kls)

            self._config_registries[config_kls] = registry

        return registry

    @property
    def wall_watch(self) -> Stopwatch:
        return self._wall_watch


_default_context: RuntimeContext | None = None


def set_runtime_context(context: RuntimeContext) -> None:
    global _default_context

    _default_context = context


def get_runtime_context() -> RuntimeContext:
    if _default_context is None:
        raise RuntimeError(
            "fairseq2 is not initialized. Make sure to call `fairseq2.setup_fairseq2()`."
        )

    return _default_context
