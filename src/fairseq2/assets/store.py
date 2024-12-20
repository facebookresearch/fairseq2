# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Protocol, TypeAlias, final

from typing_extensions import override

from fairseq2.assets.card import AssetCard
from fairseq2.assets.error import AssetCardError, AssetCardNotFoundError
from fairseq2.assets.metadata_provider import (
    AssetMetadataNotFoundError,
    AssetMetadataProvider,
    FileAssetMetadataProvider,
    PackageAssetMetadataProvider,
    WheelPackageFileLister,
)
from fairseq2.error import ContractError
from fairseq2.extensions import run_extensions
from fairseq2.utils.env import get_path_from_env
from fairseq2.utils.file import StandardFileSystem
from fairseq2.utils.yaml import load_yaml

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

    env_resolvers: list[EnvironmentResolver]
    metadata_providers: list[AssetMetadataProvider]
    user_metadata_providers: list[AssetMetadataProvider]

    def __init__(self) -> None:
        self.env_resolvers = []
        self.metadata_providers = []
        self.user_metadata_providers = []

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
                    "`envs` must be `None` since `name` already contains an environment tag."
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

        for resolver in reversed(self.env_resolvers):
            if env := resolver():
                envs.append(env)

        return envs

    def _do_retrieve_card(
        self, name: str, envs: Sequence[str], scope: str
    ) -> AssetCard:
        try:
            metadata = self._get_metadata(f"{name}@", scope)
        except AssetMetadataNotFoundError:
            raise AssetCardNotFoundError(
                f"An asset card with name '{name}' is not found."
            ) from None

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
                    f"A transitive base asset card with name '{name}' is not found."
                ) from None

        base_path = metadata.get("__base_path__")
        if base_path is not None and not isinstance(base_path, Path):
            raise contract_error("__base_path__", base_path, Path)

        metadata["name"] = name

        return AssetCard(name, metadata, base_card, base_path)

    def _get_metadata(self, name: str, scope: str) -> dict[str, object]:
        if scope == "all" or scope == "user":
            for provider in reversed(self.user_metadata_providers):
                try:
                    return provider.get_metadata(name)
                except AssetMetadataNotFoundError:
                    continue

        if scope == "all" or scope == "global":
            for provider in reversed(self.metadata_providers):
                try:
                    return provider.get_metadata(name)
                except AssetMetadataNotFoundError:
                    continue

        if name[-1] == "@":
            name = name[:-1]

        raise AssetMetadataNotFoundError(
            f"An asset metadata with name '{name}' is not found."
        ) from None

    @override
    def retrieve_names(self, *, scope: AssetScope = "all") -> list[str]:
        if scope not in ("all", "global", "user"):
            raise ValueError(
                f"`scope` must be 'all', 'global', or 'user', but is '{scope}' instead."
            )

        names = []

        if scope == "all" or scope == "user":
            for provider in self.user_metadata_providers:
                names.extend(provider.get_names())

        if scope == "all" or scope == "global":
            for provider in self.metadata_providers:
                names.extend(provider.get_names())

        return names

    def clear_cache(self) -> None:
        """Clear the cache of the underlying metadata providers."""
        for provider in self.metadata_providers:
            provider.clear_cache()

        for provider in self.user_metadata_providers:
            provider.clear_cache()

    def add_file_metadata_provider(self, path: Path, user: bool = False) -> None:
        """Add a new :class:`FileAssetMetadataProvider` pointing to ``path``.

        :param path: The directory under which asset metadata is stored.
        :param user: If ``True``, adds the metadata provider to the user scope.
        """
        file_system = StandardFileSystem()

        provider = FileAssetMetadataProvider(path, file_system, load_yaml)

        providers = self.user_metadata_providers if user else self.metadata_providers

        providers.append(provider)

    def add_package_metadata_provider(self, package_name: str) -> None:
        """Add a new :class:`PackageAssetMetadataProvider` for ``package_name``.

        :param package_name: The name of the package in which asset metadata is
            stored.
        """
        file_lister = WheelPackageFileLister()

        provider = PackageAssetMetadataProvider(package_name, file_lister, load_yaml)

        self.metadata_providers.append(provider)


class EnvironmentResolver(Protocol):
    """Resolves the environment within which assets should be loaded."""

    def __call__(self) -> str | None:
        ...


default_asset_store = StandardAssetStore()


def setup_asset_store(store: StandardAssetStore) -> None:
    store.add_package_metadata_provider("fairseq2.assets.cards")

    # /etc/fairseq2/assets
    _add_etc_dir_metadata_provider(store)

    # ~/.config/fairseq2/assets
    _add_home_config_dir_metadata_provider(store)

    # Extensions
    run_extensions("setup_fairseq2_asset_store", store)


def _add_etc_dir_metadata_provider(store: StandardAssetStore) -> None:
    asset_dir = get_path_from_env("FAIRSEQ2_ASSET_DIR")
    if asset_dir is None:
        asset_dir = Path("/etc/fairseq2/assets").resolve()
        if not asset_dir.exists():
            return

    store.add_file_metadata_provider(asset_dir)


def _add_home_config_dir_metadata_provider(store: StandardAssetStore) -> None:
    asset_dir = get_path_from_env("FAIRSEQ2_USER_ASSET_DIR")
    if asset_dir is None:
        asset_dir = get_path_from_env("XDG_CONFIG_HOME")
        if asset_dir is None:
            asset_dir = Path("~/.config").expanduser()

        asset_dir = asset_dir.joinpath("fairseq2/assets").resolve()
        if not asset_dir.exists():
            return

    store.add_file_metadata_provider(asset_dir, user=True)
