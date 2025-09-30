# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Sequence, Set
from typing import Protocol, final, runtime_checkable

from typing_extensions import override

from fairseq2.assets.card import AssetCard, AssetCardError
from fairseq2.assets.metadata_provider import AssetMetadataError, AssetMetadataProvider
from fairseq2.error import InternalError
from fairseq2.runtime.dependency import DependencyResolver, get_dependency_resolver


def get_asset_store() -> AssetStore:
    return get_dependency_resolver().resolve(AssetStore)


class AssetStore(ABC):
    @abstractmethod
    def retrieve_card(self, name: str) -> AssetCard: ...

    @abstractmethod
    def maybe_retrieve_card(self, name: str) -> AssetCard | None: ...

    @abstractmethod
    def find_cards(self, field: str, value: object) -> Iterator[AssetCard]: ...

    @property
    @abstractmethod
    def asset_names(self) -> Set[str]: ...


class AssetNotFoundError(Exception):
    def __init__(self, name: str) -> None:
        super().__init__(f"{name} asset is not found.")

        self.name = name


@final
class StandardAssetStore(AssetStore):
    def __init__(
        self,
        metadata_providers: Iterable[AssetMetadataProvider],
        *,
        default_env: str | None = None,
    ) -> None:
        if default_env is not None:
            default_env = default_env.strip()

        # user is a special, always-available environment and takes precedence
        # over the default environment.
        self._envs = ["user", default_env] if default_env else ["user"]

        self._asset_names: set[str] = set()

        self._metadata_providers: list[AssetMetadataProvider] = []

        for i, provider in enumerate(metadata_providers):
            for name in provider.asset_names:
                if not name:
                    msg = (
                        f"{provider.source} asset source has an asset with empty name."
                    )

                    raise AssetMetadataError(provider.source, msg)

                for j in range(i):
                    other_provider = self._metadata_providers[j]

                    if name in other_provider.asset_names:
                        msg = f"An asset with name {name} exists in both {provider.source} and {other_provider.source}."

                        raise AssetMetadataError(provider.source, msg)

                self._asset_names.add(name)

            self._metadata_providers.append(provider)

    @override
    def retrieve_card(self, name: str) -> AssetCard:
        card = self.maybe_retrieve_card(name)
        if card is None:
            raise AssetNotFoundError(name)

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
                msg = f"{name} base asset of the {leaf_name} asset card does not exist."

                raise AssetCardError(leaf_name, msg)

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
                f"{name} is in `asset_names`, but not in `metadata_providers`."
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
                    f"{name} is in `asset_names`, but `maybe_retrieve_card()` returned `None`."
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
class AssetEnvironmentDetector:
    def __init__(
        self,
        env_resolvers: Sequence[AssetEnvironmentResolver],
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
