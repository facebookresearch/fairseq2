# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, final

from fairseq2.assets.card import AssetCard, AssetCardError
from fairseq2.assets.metadata_provider import (
    AssetMetadataProvider,
    AssetNotFoundError,
    FileAssetMetadataProvider,
)
from fairseq2.assets.utils import _get_path_from_env
from fairseq2.typing import finaloverride


class AssetStore(ABC):
    """Represents a store of assets."""

    @abstractmethod
    def retrieve_card(self, name: str) -> AssetCard:
        """Retrieve the card of the specified asset.

        :param name:
            The name of the asset.
        """


@final
class ProviderBackedAssetStore(AssetStore):
    """Represents a store of assets backed by metadata providers."""

    env_resolvers: List[EnvironmentResolver]
    metadata_providers: List[AssetMetadataProvider]
    user_metadata_providers: List[AssetMetadataProvider]

    def __init__(self, metadata_provider: AssetMetadataProvider) -> None:
        """
        :param storage:
            The default asset metadata provider.
        """
        self.env_resolvers = []
        self.metadata_providers = [metadata_provider]
        self.user_metadata_providers = []

    @finaloverride
    def retrieve_card(self, name: str) -> AssetCard:
        if "@" in name:
            raise ValueError("`name` must not contain the reserved '@' character.")

        envs = self._resolve_envs()

        return self._do_retrieve_card(name, envs)

    def _resolve_envs(self) -> List[str]:
        envs = []

        for resolver in self.env_resolvers:
            if env := resolver():
                envs.append(env)

        # This is a special, always available environment for users to override
        # asset metadata. For instance, a user can set the checkpoint path of a
        # gated model locally by having a same named asset with @user suffix.
        envs.append("user")

        return envs

    def _do_retrieve_card(self, name: str, envs: List[str]) -> AssetCard:
        metadata = self._get_metadata(name)

        # If we have environment-specific metadata, merge it with `metadata`.
        for env in envs:
            try:
                env_metadata = self._get_metadata(f"{name}@{env}")

                # Do not allow overriding 'name'.
                del env_metadata["name"]

                metadata.update(env_metadata)
            except AssetNotFoundError:
                pass

        try:
            base_name = metadata["base"]
        except KeyError:
            base_name = None

        base_card: Optional[AssetCard] = None

        # If the metadata has a base specified, we have to recursively load the
        # entire chain up to the root.
        if base_name:
            if not isinstance(base_name, str):
                raise AssetCardError(
                    f"The value of the field 'base' of the asset card '{name}' must be of type `{str}`, but is of type `{type(base_name)}` instead."
                )

            base_card = self._do_retrieve_card(base_name, envs)

        return AssetCard(metadata, base_card)

    def _get_metadata(self, name: str) -> Dict[str, Any]:
        for provider in reversed(self.user_metadata_providers):
            try:
                return provider.get_metadata(name)
            except AssetNotFoundError:
                continue

        for provider in reversed(self.metadata_providers):
            try:
                return provider.get_metadata(name)
            except AssetNotFoundError:
                continue

        raise AssetNotFoundError(f"An asset with the name '{name}' cannot be found.")

    def clear_cache(self) -> None:
        """Clear the cache of the underlying metadata providers."""
        for provider in self.metadata_providers:
            provider.clear_cache()

        for provider in self.user_metadata_providers:
            provider.clear_cache()


class EnvironmentResolver(Protocol):
    """Resolves the environment within which assets should be loaded.

    Assets can have varying metadata depending on the environment that they are
    loaded in due to regulatory or technical requirements.
    """

    def __call__(self) -> Optional[str]:
        ...


def _create_asset_store() -> ProviderBackedAssetStore:
    cards_dir = Path(__file__).parent.joinpath("cards")

    metadata_provider = FileAssetMetadataProvider(cards_dir)

    return ProviderBackedAssetStore(metadata_provider)


asset_store = _create_asset_store()


def _load_asset_directory() -> None:
    asset_dir = _get_path_from_env("FAIRSEQ2_ASSET_DIR")
    if asset_dir is None:
        asset_dir = Path("/etc/fairseq2/assets").resolve()
        if not asset_dir.exists():
            return

    asset_store.metadata_providers.append(FileAssetMetadataProvider(asset_dir))


_load_asset_directory()


def _load_user_asset_directory() -> None:
    asset_dir = _get_path_from_env("FAIRSEQ2_USER_ASSET_DIR")
    if asset_dir is None:
        asset_dir = _get_path_from_env("XDG_CONFIG_HOME")
        if asset_dir is None:
            asset_dir = Path("~/.config").expanduser()

        asset_dir = asset_dir.joinpath("fairseq2/assets").resolve()
        if not asset_dir.exists():
            return

    asset_store.user_metadata_providers.append(FileAssetMetadataProvider(asset_dir))


_load_user_asset_directory()


# TODO: Move to fairseq2-ext.
def _load_faircluster() -> None:
    if "FAIR_ENV_CLUSTER" not in os.environ:
        return

    asset_store.env_resolvers.append(lambda: "faircluster")

    # This directory is meant to store cluster-wide asset cards.
    asset_dir = Path("/checkpoint/balioglu/fairseq2-ext/cards")
    if asset_dir.exists():
        asset_store.metadata_providers.append(FileAssetMetadataProvider(asset_dir))


_load_faircluster()
