# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, TypeVar, final

from fairseq2.assets import AssetDownloadManager, StandardAssetStore
from fairseq2.config_registry import ConfigRegistry
from fairseq2.registry import Registry

T = TypeVar("T")


@final
class RuntimeContext:
    _asset_store: StandardAssetStore
    _asset_download_manager: AssetDownloadManager
    _registries: dict[type, Registry[Any]]
    _config_registries: dict[type, ConfigRegistry[Any]]

    def __init__(
        self,
        asset_store: StandardAssetStore,
        asset_download_manager: AssetDownloadManager,
    ) -> None:
        self._asset_store = asset_store
        self._asset_download_manager = asset_download_manager

        self._registries = {}
        self._config_registries = {}

    @property
    def asset_store(self) -> StandardAssetStore:
        return self._asset_store

    @property
    def asset_download_manager(self) -> AssetDownloadManager:
        return self._asset_download_manager

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
