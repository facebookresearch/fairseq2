# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Iterable, Literal, Protocol, TypeAlias, final, runtime_checkable

from typing_extensions import override

from fairseq2.assets.card import AssetCard, AssetCardError
from fairseq2.assets.metadata_provider import (
    AssetMetadataProvider,
    AssetNotFoundError,
    FileAssetMetadataProvider,
    PackageAssetMetadataProvider,
)
from fairseq2.dependency import DependencyContainer, resolve
from fairseq2.utils.structured import ValueConverter

AssetScope: TypeAlias = Literal["all", "global", "user"]


class AssetStore(ABC):
    """Represents a store of assets."""

    @abstractmethod
    def retrieve_card(
        self, name: str, *, envs: Sequence[str] | None = None, scope: AssetScope = "all"
    ) -> AssetCard:
        """Retrieve the card of the specified asset.

        :param name:
            The name of the asset.
        :para env:
            The environments, in order of precedence, in which to retrieve the
            card. If ``None``, the available environments will be resolved
            automatically.
        :param scope:
            The scope of retrieval.
        """

    @abstractmethod
    def retrieve_names(self, *, scope: AssetScope = "all") -> list[str]:
        """Retrieve the names of the assets contained in this store.

        :param scope:
            The scope of retrieval.
        """


@final
class StandardAssetStore(AssetStore):
    """Represents a store of assets."""

    _env_resolvers: list[EnvironmentResolver]
    _metadata_providers: list[AssetMetadataProvider]
    _user_metadata_providers: list[AssetMetadataProvider]
    _value_converter: ValueConverter

    def __init__(
        self,
        env_resolvers: Iterable[EnvironmentResolver],
        metadata_providers: Iterable[AssetMetadataProvider],
        value_converter: ValueConverter,
    ) -> None:
        self._env_resolvers = list(env_resolvers)
        self._metadata_providers = []
        self._user_metadata_providers = []
        self._value_converter = value_converter

        for idx, metadata_provider in enumerate(metadata_providers):
            if metadata_provider.scope == "global":
                self._metadata_providers.append(metadata_provider)

                continue

            if metadata_provider.scope == "user":
                self._user_metadata_providers.append(metadata_provider)

                continue

            raise ValueError(
                f"The scope of a `MetadataProvider` must be 'global' or 'user', but the instance at index {idx} in `metadata_providers` has an unsupported scope '{metadata_provider.scope}' instead."
            )

    @override
    def retrieve_card(
        self, name: str, *, envs: Sequence[str] | None = None, scope: AssetScope = "all"
    ) -> AssetCard:
        if scope not in ("all", "global", "user"):
            raise ValueError(
                f"`scope` must be 'all', 'global', or 'user', but is '{scope}' instead."
            )

        name_env_pair = name.split("@", maxsplit=1)

        name = name_env_pair[0]

        # See if we have an environment tag.
        if len(name_env_pair) == 2:
            if envs is not None:
                raise ValueError(
                    "`name` already contains an environment tag, `envs` must be `None`."
                )

            envs = [name_env_pair[1]]

        if envs is None:
            envs = self._resolve_envs()

        return self._do_retrieve_card(name, envs, scope)

    def _resolve_envs(self) -> list[str]:
        # This is a special, always available environment for users to override
        # asset metadata. For instance, a user can set the checkpoint path of a
        # gated model locally by having a same-named asset with a @user suffix.
        envs = ["user"]

        for resolver in reversed(self._env_resolvers):
            if env := resolver():
                envs.append(env)

        return envs

    def _do_retrieve_card(
        self, name: str, envs: Sequence[str], scope: str
    ) -> AssetCard:
        metadata = self._get_metadata(f"{name}@", scope)

        # If we have environment-specific metadata, merge it with `metadata`.
        for env in reversed(envs):
            try:
                env_metadata = self._get_metadata(f"{name}@{env}", scope)

                # Do not allow overriding 'name'.
                try:
                    del env_metadata["name"]
                except KeyError:
                    pass

                metadata.update(env_metadata)
            except AssetNotFoundError:
                pass

        try:
            base_name = metadata["base"]
        except KeyError:
            base_name = None

        base_card: AssetCard | None = None

        # If the metadata has a base specified, we have to recursively load the
        # entire chain up to the root.
        if base_name:
            if not isinstance(base_name, str):
                raise AssetCardError(
                    f"The value of the field 'base' of the asset card '{name}' must be of type `{str}`, but is of type `{type(base_name)}` instead."
                )

            base_card = self._do_retrieve_card(base_name, envs, scope)

        metadata["name"] = name

        return AssetCard(metadata, base_card, self._value_converter)

    def _get_metadata(self, name: str, scope: str) -> dict[str, Any]:
        if scope == "all" or scope == "user":
            for provider in reversed(self._user_metadata_providers):
                try:
                    return provider.get_metadata(name)
                except AssetNotFoundError:
                    continue

        if scope == "all" or scope == "global":
            for provider in reversed(self._metadata_providers):
                try:
                    return provider.get_metadata(name)
                except AssetNotFoundError:
                    continue

        if name[-1] == "@":
            name = name[:-1]

        raise AssetNotFoundError(
            name, f"An asset with the name '{name}' cannot be found. Run `fairseq2 assets list` to see the list of available assets."  # fmt: skip
        )

    @override
    def retrieve_names(self, *, scope: AssetScope = "all") -> list[str]:
        if scope not in ("all", "global", "user"):
            raise ValueError(
                f"`scope` must be 'all', 'global', or 'user', but is '{scope}' instead."
            )

        names = []

        if scope == "all" or scope == "user":
            for provider in self._user_metadata_providers:
                names.extend(provider.get_names())

        if scope == "all" or scope == "global":
            for provider in self._metadata_providers:
                names.extend(provider.get_names())

        return names

    def clear_cache(self) -> None:
        """Clear the cache of the underlying metadata providers."""
        for provider in self._metadata_providers:
            provider.clear_cache()

        for provider in self._user_metadata_providers:
            provider.clear_cache()


@runtime_checkable
class EnvironmentResolver(Protocol):
    """Resolves the environment within which assets should be loaded."""

    def __call__(self) -> str | None:
        ...


def register_asset_store(container: DependencyContainer) -> None:
    container.register(StandardAssetStore)

    container.register_factory(AssetStore, lambda r: r.resolve(StandardAssetStore))


def get_asset_store() -> AssetStore:
    return resolve(AssetStore)  # type: ignore[no-any-return]


# COMPAT


class _SingletonAssetStore(AssetStore):
    @override
    def retrieve_card(
        self, name: str, *, envs: Sequence[str] | None = None, scope: AssetScope = "all"
    ) -> AssetCard:
        return resolve(StandardAssetStore).retrieve_card(name, envs=envs, scope=scope)

    @override
    def retrieve_names(self, *, scope: AssetScope = "all") -> list[str]:
        return resolve(StandardAssetStore).retrieve_names(scope=scope)

    def add_file_metadata_provider(self, path: Path, user: bool = False) -> None:
        providers = self.user_metadata_providers if user else self.metadata_providers

        providers.append(FileAssetMetadataProvider(path))

    def add_package_metadata_provider(self, package_name: str) -> None:
        self.metadata_providers.append(PackageAssetMetadataProvider(package_name))

    @property
    def env_resolvers(self) -> list[EnvironmentResolver]:
        return resolve(StandardAssetStore)._env_resolvers  # type: ignore[return-value]

    @property
    def metadata_providers(self) -> list[AssetMetadataProvider]:
        return resolve(StandardAssetStore)._metadata_providers

    @property
    def user_metadata_providers(self) -> list[AssetMetadataProvider]:
        return resolve(StandardAssetStore)._user_metadata_providers


default_asset_store = _SingletonAssetStore()
