# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Hashable, Iterable
from typing import Any, Generic, Mapping, TypeVar, final

from typing_extensions import override

from fairseq2.assets import AssetDownloadManager, StandardAssetStore
from fairseq2.error import AlreadyExistsError

T = TypeVar("T")


@final
class RuntimeContext:
    _asset_store: StandardAssetStore
    _asset_download_manager: AssetDownloadManager
    _registries: Mapping[type, Registry[Any]]

    def __init__(
        self,
        asset_store: StandardAssetStore,
        asset_download_manager: AssetDownloadManager,
    ) -> None:
        self._asset_store = asset_store
        self._asset_download_manager = asset_download_manager

        self._registries = defaultdict(Registry)

    @property
    def asset_store(self) -> StandardAssetStore:
        return self._asset_store

    @property
    def asset_download_manager(self) -> AssetDownloadManager:
        return self._asset_download_manager

    def get_registry(self, kls: type[T]) -> Registry[T]:
        return self._registries[kls]


T_co = TypeVar("T_co", covariant=True)


class Provider(ABC, Generic[T_co]):
    @abstractmethod
    def get(self, key: Hashable) -> T_co:
        ...

    @abstractmethod
    def get_all(self) -> Iterable[tuple[Hashable, T_co]]:
        ...


@final
class Registry(Provider[T]):
    _entries: dict[Hashable, T]

    def __init__(self) -> None:
        self._entries = {}

    @override
    def get(self, key: Hashable) -> T:
        try:
            return self._entries[key]
        except KeyError:
            raise LookupError(f"The registry does not contain a '{key}' key.") from None

    @override
    def get_all(self) -> Iterable[tuple[Hashable, T]]:
        return self._entries.items()

    def register(self, key: Hashable, value: T) -> None:
        if key in self._entries:
            raise AlreadyExistsError(f"The registry already contains a '{key}' key.")

        self._entries[key] = value


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
