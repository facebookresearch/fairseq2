# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Set
from typing import Protocol, final, runtime_checkable

from typing_extensions import override

from fairseq2.assets.card import AssetCard
from fairseq2.assets.metadata_provider import (
    AssetMetadataProvider,
    BadAssetMetadataError,
)
from fairseq2.error import InternalError
from fairseq2.runtime.dependency import DependencyResolver, get_dependency_resolver


def get_asset_store() -> AssetStore:
    return get_dependency_resolver().resolve(AssetStore)


class AssetStore(ABC):
    @abstractmethod
    def retrieve_card(self, name: str) -> AssetCard:
        """
        :raises AssetCardNotFoundError:
        :raises BaseAssetCardNotFoundError:
        :raises AssetStoreError:
        """

    @abstractmethod
    def maybe_retrieve_card(self, name: str) -> AssetCard | None:
        """
        :raises BaseAssetCardNotFoundError:
        :raises AssetStoreError:
        """

    @abstractmethod
    def find_cards(self, field: str, value: object) -> Iterator[AssetCard]:
        """
        :raises AssetStoreError:
        """

    @property
    @abstractmethod
    def asset_names(self) -> Set[str]: ...


class AssetStoreError(Exception):
    pass


class AssetCardNotFoundError(AssetStoreError):
    def __init__(self, name: str) -> None:
        super().__init__(f"an asset card named '{name}' is not found")

        self.name = name


class BaseAssetCardNotFoundError(AssetStoreError):
    def __init__(self, name: str, base_name: str) -> None:
        super().__init__(f"base asset card '{base_name}' of '{name}' is not found")

        self.name = name
        self.base_name = base_name


@final
class _StandardAssetStore(AssetStore):
    def __init__(
        self,
        metadata_providers: Iterable[AssetMetadataProvider],
        *,
        default_env: str | None = None,
    ) -> None:
        """
        :raises CorruptAssetMetadataError:
        """
        if default_env is not None:
            default_env = default_env.strip()

        # user is a special, always-available environment and takes precedence
        # over the default environment.
        envs = ["user", default_env] if default_env else ["user"]

        asset_names: set[str] = set()

        metadata_provider_list: list[AssetMetadataProvider] = []

        for i, provider in enumerate(metadata_providers):
            for name in provider.asset_names:
                if not name:
                    raise BadAssetMetadataError(
                        provider.source, f"asset metadata source '{provider.source}' has an asset with empty name"  # fmt: skip
                    )

                for j in range(i):
                    other_provider = metadata_provider_list[j]

                    if name in other_provider.asset_names:
                        raise BadAssetMetadataError(
                            provider.source, f"an asset with name '{name}' exists in both asset metadata sources '{provider.source}' and '{other_provider.source}'"  # fmt: skip
                        )

                asset_names.add(name)

            metadata_provider_list.append(provider)

        self._envs = envs
        self._asset_names = asset_names
        self._metadata_providers = metadata_provider_list

    @override
    def retrieve_card(self, name: str) -> AssetCard:
        card = self.maybe_retrieve_card(name)
        if card is None:
            raise AssetCardNotFoundError(name)

        return card

    @override
    def maybe_retrieve_card(self, name: str) -> AssetCard | None:
        name_env_pair = name.split("@", maxsplit=1)

        if len(name_env_pair) == 1:
            name = name_env_pair[0]

            envs = self._envs
        else:
            name, env = name_env_pair

            env = env.strip()

            # An empty environment tag (i.e. 'name@') means no environment
            # look-up should be performed.
            envs = [env] if env else []

        name = name.strip()
        if not name:
            return None

        return self._do_retrieve_card(name, name, envs)

    def _do_retrieve_card(
        self, name: str, leaf_name: str, envs: list[str]
    ) -> AssetCard | None:
        metadata = self._maybe_get_metadata(f"{name}@")
        if metadata is None:
            if name != leaf_name:
                raise BaseAssetCardNotFoundError(leaf_name, name)

            return None

        for env in envs:
            env_metadata = self._maybe_get_metadata(f"{name}@{env}")
            if env_metadata is not None:
                metadata.update(env_metadata)

                break

        # Load the base card.
        base_name = metadata.get("base")
        if not isinstance(base_name, str):
            base_card = None
        else:
            base_card = self._do_retrieve_card(base_name, leaf_name, envs)

        return AssetCard(name, metadata, base_card)

    def _maybe_get_metadata(self, name: str) -> dict[str, object] | None:
        for provider in self._metadata_providers:
            metadata = provider.maybe_get_metadata(name)
            if metadata is None:
                continue

            metadata["__source__"] = provider.source

            return metadata

        if name in self._asset_names:
            raise InternalError(
                f"'{name}' is in `asset_names`, but not in `metadata_providers`"
            )

        return None

    @override
    def find_cards(self, field: str, value: object) -> Iterator[AssetCard]:
        for name in self._asset_names:
            if name[-1] != "@":  # skip environment-specific assets
                continue

            card = self.maybe_retrieve_card(name[:-1])
            if card is None:
                raise InternalError(
                    f"'{name}' is in `asset_names`, but `maybe_retrieve_card()` returned `None`"
                )

            field_ = card.maybe_get_field(field)
            if field_ is not None:
                if value == field_.value:
                    yield card

    @property
    @override
    def asset_names(self) -> Set[str]:
        return self._asset_names


@runtime_checkable
class AssetEnvironmentResolver(Protocol):
    def __call__(self, resolver: DependencyResolver) -> str | None: ...


@final
class _AssetEnvironmentDetector:
    def __init__(
        self,
        env_resolvers: Iterable[AssetEnvironmentResolver],
        resolver: DependencyResolver,
    ) -> None:
        self._env_resolvers = env_resolvers
        self._resolver = resolver

    def detect(self) -> str | None:
        for env_resolver in self._env_resolvers:
            env = env_resolver(self._resolver)
            if env is not None:
                return env

        return None
