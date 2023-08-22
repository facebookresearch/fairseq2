# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, final

from fairseq2.assets.card import AssetCard, AssetCardError
from fairseq2.assets.card_storage import (
    AssetCardNotFoundError,
    AssetCardStorage,
    LocalAssetCardStorage,
)
from fairseq2.typing import finaloverride


class AssetStore(ABC):
    """Provides access to asset cards stored in a centralized location."""

    env: Optional[str]
    """
    An environment is a mechanism to conditionally override fields of an asset
    card. It is typically used to replace asset information that differs in a
    specific environment due to regulatory or technical reasons (e.g. checkpoint
    locations in a cluster with no internet access).

    If not ``None``, :class:`AssetStore` will check if there is an environment-
    specific asset card for ``env`` and, if one is found, will merge its content
    with the generic asset card.
    """

    @abstractmethod
    def retrieve_card(self, name: str, ignore_cache: bool = False) -> AssetCard:
        """Retrieve the card of the specified asset.

        :param name:
            The name of the asset.
        :param ignore_cache:
            If ``True``, retrieves the asset card from the storage even if it is
            already in cache.
        """

    @abstractmethod
    def register_card(self, card: AssetCard, env: Optional[str] = None) -> None:
        """Register the specified asset card.

        :param card:
            The asset card.
        :param env:
            If not ``None``, registers as an environment-specific asset card.
        """

    @abstractmethod
    def clear_cache(self) -> None:
        """Clear the in-memory :class:`AssetCard` cache."""


@final
class DefaultAssetStore(AssetStore):
    """Provides access to asset cards stored in a centralized location."""

    _storage: AssetCardStorage
    _cache: Dict[str, AssetCard]

    def __init__(self, storage: AssetCardStorage, ignore_env: bool = False) -> None:
        """
        :param storage:
            The asset card storage to use.
        :param ignore_env:
            If ``True``, ignores environment-specific asset cards.
        """
        if ignore_env:
            self.env = None
        else:
            self.env = self._determine_environment()

        self._storage = storage

        self._cache = {}

    @staticmethod
    def _determine_environment() -> Optional[str]:
        # TODO: Make extensible instead of hard-coded conditions.
        if "FAIR_ENV_CLUSTER" in os.environ:
            return "faircluster"

        return None

    @finaloverride
    def retrieve_card(self, name: str, ignore_cache: bool = False) -> AssetCard:
        if not ignore_cache:
            try:
                return self._cache[name]
            except KeyError:
                pass

        data = self._storage.load_card(name)

        if self.env:
            try:
                env_data = self._storage.load_card(name, self.env)

                # If we have an environment-specific asset card, merge it with
                # the generic one.
                data.update(env_data)
            except AssetCardNotFoundError:
                pass

        try:
            base_name = data["base"]
        except KeyError:
            base_name = None

        base: Optional[AssetCard] = None

        # If the asset card has a base specified, we have to recursively load
        # the entire chain up to the root card.
        if base_name:
            if not isinstance(base_name, str):
                raise AssetCardError(
                    f"The type of the field 'base' of the asset card '{name}' must be `{str}`, but is `{type(base_name)}` instead."
                )

            base = self.retrieve_card(base_name, ignore_cache)

        card = AssetCard(name, data, base)

        self._cache[name] = card

        return card

    @finaloverride
    def register_card(self, card: AssetCard, env: Optional[str] = None) -> None:
        raise NotImplementedError()

    @finaloverride
    def clear_cache(self) -> None:
        self._cache.clear()


def create_default_asset_store() -> AssetStore:
    pathname = Path(__file__).parent.joinpath("cards")

    card_storage = LocalAssetCardStorage(pathname)

    return DefaultAssetStore(card_storage)


asset_store = create_default_asset_store()
