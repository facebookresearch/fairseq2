# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence, Set
from typing import Protocol, final, runtime_checkable

from typing_extensions import override

from fairseq2.assets.card import AssetCard, AssetCardError, AssetCardFieldNotFoundError
from fairseq2.assets.dirs import AssetDirectories
from fairseq2.assets.metadata_provider import (
    AssetMetadataError,
    AssetMetadataProvider,
    AssetNotFoundError,
    FileBackedAssetMetadataLoader,
    YamlAssetMetadataFileLoader,
    _load_package_assets,
)
from fairseq2.error import ContractError, InternalError
from fairseq2.file_system import FileSystem
from fairseq2.runtime.dependency import DependencyResolver, get_dependency_resolver
from fairseq2.utils.env import get_env
from fairseq2.utils.yaml import YamlLoader


class AssetStore(ABC):
    """Represents a store of assets."""

    @abstractmethod
    def retrieve_card(self, name: str) -> AssetCard:
        """Retrieve the card of the specified asset."""

    @abstractmethod
    def find_cards(self, field: str, value: object) -> Iterable[AssetCard]: ...

    @property
    @abstractmethod
    def asset_names(self) -> Set[str]:
        """Gets the names of the assets contained in this store."""


@final
class StandardAssetStore(AssetStore):
    """Represents a store of assets."""

    _envs: list[str]
    _asset_names: set[str]
    _metadata_providers: Sequence[AssetMetadataProvider]

    def __init__(
        self,
        metadata_providers: Sequence[AssetMetadataProvider],
        *,
        default_env: str | None = None,
    ) -> None:
        if default_env is not None:
            default_env = default_env.strip()

        # 'user' is a special, always-available environment and takes precedence
        # over the default environment.
        self._envs = ["user", default_env] if default_env else ["user"]

        self._asset_names = set()

        for i, provider in enumerate(metadata_providers):
            for name in provider.asset_names:
                for j in range(i):
                    other_provider = metadata_providers[j]

                    if name in other_provider.asset_names:
                        raise AssetMetadataError(
                            provider.source, f"An asset with name '{name}' exists in both '{provider.source}' and '{other_provider.source}'."  # fmt: skip
                        )

                self._asset_names.add(name)

        self._metadata_providers = metadata_providers

    @override
    def retrieve_card(self, name: str) -> AssetCard:
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

        try:
            return self._do_retrieve_card(name.strip(), envs)
        except AssetNotFoundError as ex:
            if ex.name != name:
                raise AssetCardError(
                    name, f"A base asset with name '{ex.name}' is not found."
                ) from None

            raise

    def _do_retrieve_card(self, name: str, envs: list[str]) -> AssetCard:
        metadata = self._get_metadata(f"{name}@")

        for env in envs:
            try:
                env_metadata = self._get_metadata(f"{name}@{env}")
            except AssetNotFoundError:
                continue

            metadata.update(env_metadata)

            break

        # Load the base card.
        base_name = metadata.get("base")
        if base_name is None:
            base_card = None
        else:
            if not isinstance(base_name, str):
                raise ContractError(
                    f"The value of the 'base' field of the '{name}' asset card is expected to be of type `{str}`, but is of type `{type(base_name)}` instead."
                )

            base_card = self._do_retrieve_card(base_name, envs)

        return AssetCard(name, metadata, base_card)

    def _get_metadata(self, name: str) -> dict[str, object]:
        for provider in self._metadata_providers:
            if name in provider.asset_names:
                try:
                    metadata = provider.get_metadata(name)
                except AssetNotFoundError:
                    raise ContractError(
                        f"The '{provider.source}' asset source does not have an asset named '{name}'."
                    ) from None

                metadata["__source__"] = provider.source

                return metadata

        if name[-1] == "@":
            name = name[:-1]

        raise AssetNotFoundError(name)

    @override
    def find_cards(self, field: str, value: object) -> Iterable[AssetCard]:
        for name in self._asset_names:
            if name[-1] != "@":  # skip environment-specific assets
                continue

            try:
                card = self.retrieve_card(name[:-1])
            except AssetNotFoundError:
                raise InternalError(
                    f"'{name}' is in `self.asset_names`, but cannot be found in the store."
                ) from None

            try:
                field_value = card.field(field).value()
            except AssetCardFieldNotFoundError:
                continue

            if field_value == value:
                yield card

    @property
    @override
    def asset_names(self) -> Set[str]:
        return self._asset_names


@runtime_checkable
class AssetEnvironmentResolver(Protocol):
    """Resolves the environment within which assets should be loaded."""

    def __call__(self, resolver: DependencyResolver) -> str | None: ...


def _load_asset_store(resolver: DependencyResolver) -> AssetStore:
    env_resolvers = resolver.resolve_all(AssetEnvironmentResolver)

    other_metadata_providers = resolver.resolve_all(AssetMetadataProvider)

    file_system = resolver.resolve(FileSystem)

    yaml_loader = resolver.resolve(YamlLoader)

    env = get_env(resolver)

    asset_dirs = AssetDirectories(env, file_system)

    metadata_file_loader = YamlAssetMetadataFileLoader(yaml_loader)

    metadata_providers = []

    metadata_provider: AssetMetadataProvider | None

    # Package
    metadata_provider = _load_package_assets(
        resolver, package_name="fairseq2.assets.cards"
    )

    metadata_providers.append(metadata_provider)

    # System
    metadata_dir = asset_dirs.get_system_dir()
    if metadata_dir is not None:
        metadata_loader = FileBackedAssetMetadataLoader(
            file_system, metadata_file_loader
        )

        try:
            metadata_provider = metadata_loader.load(metadata_dir)
        except FileNotFoundError:
            metadata_provider = None

        if metadata_provider is not None:
            metadata_providers.append(metadata_provider)

    # User
    metadata_dir = asset_dirs.get_user_dir()
    if metadata_dir is not None:
        metadata_loader = FileBackedAssetMetadataLoader(
            file_system, metadata_file_loader
        )

        try:
            metadata_provider = metadata_loader.load(metadata_dir)
        except FileNotFoundError:
            metadata_provider = None

        if metadata_provider is not None:
            metadata_providers.append(metadata_provider)

    metadata_providers.extend(other_metadata_providers)

    asset_env = None

    for env_resolver in env_resolvers:
        asset_env = env_resolver(resolver)
        if asset_env is not None:
            break

    return StandardAssetStore(metadata_providers, default_env=asset_env)


def get_asset_store() -> AssetStore:
    return get_dependency_resolver().resolve(AssetStore)
