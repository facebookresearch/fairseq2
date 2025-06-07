# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from enum import Enum
from pathlib import Path
from typing import Protocol, final, runtime_checkable

from typing_extensions import override

from fairseq2.assets.card import AssetCard, AssetCardError, AssetCardNotFoundError
from fairseq2.assets.dirs import AssetDirectories
from fairseq2.assets.metadata_provider import (
    AssetMetadataLoadError,
    AssetMetadataNotFoundError,
    AssetMetadataProvider,
    FileBackedAssetMetadataLoader,
    YamlAssetMetadataFileLoader,
    load_package_asset_provider,
)
from fairseq2.dependency import DependencyResolver
from fairseq2.error import ContractError, SetupError
from fairseq2.file_system import FileSystem
from fairseq2.utils.env import get_env
from fairseq2.utils.yaml import YamlLoader


class AssetLookupScope(Enum):
    ALL = 0
    SYSTEM = 1
    USER = 2


class AssetStore(ABC):
    """Represents a store of assets."""

    @abstractmethod
    def retrieve_card(
        self,
        name: str,
        *,
        envs: Sequence[str] | None = None,
        scope: AssetLookupScope = AssetLookupScope.ALL,
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
    def retrieve_names(
        self, *, scope: AssetLookupScope = AssetLookupScope.ALL
    ) -> list[str]:
        """Retrieve the names of the assets contained in this store.

        :param scope:
            The scope of retrieval.
        """


@final
class StandardAssetStore(AssetStore):
    """Represents a store of assets."""

    _env_resolvers: Iterable[EnvironmentResolver]
    _metadata_providers: Iterable[AssetMetadataProvider]
    _user_metadata_providers: Iterable[AssetMetadataProvider]

    def __init__(
        self,
        env_resolvers: Iterable[EnvironmentResolver],
        metadata_providers: Iterable[AssetMetadataProvider],
        user_metadata_providers: Iterable[AssetMetadataProvider],
    ) -> None:
        self._env_resolvers = env_resolvers
        self._metadata_providers = metadata_providers
        self._user_metadata_providers = user_metadata_providers

    @override
    def retrieve_card(
        self,
        name: str,
        *,
        envs: Sequence[str] | None = None,
        scope: AssetLookupScope = AssetLookupScope.ALL,
    ) -> AssetCard:
        name_env_pair = name.split("@", maxsplit=1)

        name = name_env_pair[0]

        # See if we have an environment tag.
        if len(name_env_pair) == 2:
            if envs is not None:
                raise ValueError(
                    "`envs` must not be specified since `name` already contains an environment tag."
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

        for resolver in self._env_resolvers:
            env = resolver()
            if env is not None:
                envs.append(env)

        return envs

    def _do_retrieve_card(
        self, name: str, envs: Sequence[str], scope: AssetLookupScope
    ) -> AssetCard:
        try:
            metadata = self._get_metadata(f"{name}@", scope)
        except AssetMetadataNotFoundError:
            raise AssetCardNotFoundError(name) from None

        # If we have environment-specific metadata, merge it with `metadata`.
        for env in envs:
            try:
                env_metadata = self._get_metadata(f"{name}@{env}", scope)

                # Do not allow overriding 'name'.
                try:
                    del env_metadata["name"]
                except KeyError:
                    pass

                metadata.update(env_metadata)
            except AssetMetadataNotFoundError:
                pass

        def contract_error(
            field: str, value: object, expected_kls: object
        ) -> ContractError:
            return ContractError(
                f"The value of the '{field}' field of the '{name}' asset card is expected to be of type `{expected_kls}`, but is of type `{type(value)}` instead."
            )

        base_name = metadata.get("base")

        base_card: AssetCard | None = None

        # If the metadata has a base specified, we have to recursively load the
        # entire chain up to the root.
        if base_name is not None:
            if not isinstance(base_name, str):
                raise contract_error("base", base_name, "str")

            try:
                base_card = self._do_retrieve_card(base_name, envs, scope)
            except AssetCardNotFoundError:
                raise AssetCardError(
                    name, f"A transitive base asset card with name '{base_name}' is not found."  # fmt: skip
                ) from None

        base_path = metadata.get("__base_path__")
        if base_path is not None and not isinstance(base_path, Path):
            raise contract_error("__base_path__", base_path, Path)

        metadata["name"] = name

        return AssetCard(name, metadata, base_card, base_path)

    def _get_metadata(self, name: str, scope: AssetLookupScope) -> dict[str, object]:
        if scope == AssetLookupScope.ALL or scope == AssetLookupScope.USER:
            for provider in self._user_metadata_providers:
                try:
                    return provider.get_metadata(name)
                except AssetMetadataNotFoundError:
                    continue

        if scope == AssetLookupScope.ALL or scope == AssetLookupScope.SYSTEM:
            for provider in self._metadata_providers:
                try:
                    return provider.get_metadata(name)
                except AssetMetadataNotFoundError:
                    continue

        if name[-1] == "@":
            name = name[:-1]

        raise AssetMetadataNotFoundError(name)

    @override
    def retrieve_names(
        self, *, scope: AssetLookupScope = AssetLookupScope.ALL
    ) -> list[str]:
        names = []

        if scope == AssetLookupScope.ALL or scope == AssetLookupScope.USER:
            for provider in self._user_metadata_providers:
                names.extend(provider.get_names())

        if scope == AssetLookupScope.ALL or scope == AssetLookupScope.SYSTEM:
            for provider in self._metadata_providers:
                names.extend(provider.get_names())

        return names


@runtime_checkable
class EnvironmentResolver(Protocol):
    """Resolves the environment within which assets should be loaded."""

    def __call__(self) -> str | None: ...


def load_asset_store(resolver: DependencyResolver) -> AssetStore:
    env_resolvers = resolver.resolve_all(EnvironmentResolver)

    yaml_loader = resolver.resolve(YamlLoader)

    env = get_env(resolver)

    file_system = resolver.resolve(FileSystem)

    asset_metadata_file_loader = YamlAssetMetadataFileLoader(yaml_loader)

    asset_dirs = AssetDirectories(env, file_system)

    metadata_provider: AssetMetadataProvider | None

    # System
    metadata_provider = load_package_asset_provider(
        resolver, package_name="fairseq2.assets.cards"
    )

    metadata_providers = [metadata_provider]

    try:
        config_dir = asset_dirs.get_system_dir()
    except AssetMetadataLoadError as ex:
        raise SetupError(
            "The system asset metadata directory cannot be determined. See the nested exception for details."
        ) from ex

    if config_dir is not None:
        metadata_loader = FileBackedAssetMetadataLoader(
            config_dir, file_system, asset_metadata_file_loader
        )

        try:
            metadata_provider = metadata_loader.load()
        except FileNotFoundError:
            metadata_provider = None
        except AssetMetadataLoadError as ex:
            raise SetupError(
                f"The asset metadata at the '{config_dir}' path cannot be loaded. See the nested exception for details."
            ) from ex

        if metadata_provider is not None:
            metadata_providers.append(metadata_provider)

    for metadata_provider in resolver.resolve_all(AssetMetadataProvider):
        metadata_providers.append(metadata_provider)

    # User
    user_metadata_providers = []

    try:
        config_dir = asset_dirs.get_user_dir()
    except AssetMetadataLoadError as ex:
        raise SetupError(
            "The user asset metadata directory cannot be determined. See the nested exception for details."
        ) from ex

    if config_dir is not None:
        metadata_loader = FileBackedAssetMetadataLoader(
            config_dir, file_system, asset_metadata_file_loader
        )

        try:
            metadata_provider = metadata_loader.load()
        except FileNotFoundError:
            metadata_provider = None
        except AssetMetadataLoadError as ex:
            raise SetupError(
                f"The asset metadata at the '{config_dir}' path cannot be loaded. See the nested exception for details."
            ) from ex

        if metadata_provider is not None:
            user_metadata_providers.append(metadata_provider)

    for metadata_provider in resolver.resolve_all(AssetMetadataProvider, key="user"):
        user_metadata_providers.append(metadata_provider)

    return StandardAssetStore(
        env_resolvers, metadata_providers, user_metadata_providers
    )
