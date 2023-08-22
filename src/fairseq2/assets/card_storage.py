# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, final

import yaml
from yaml import YAMLError

from fairseq2.assets.card import AssetCardError
from fairseq2.assets.error import AssetError
from fairseq2.typing import finaloverride


class AssetCardStorage(ABC):
    """Stores asset cards on a persistent storage."""

    @abstractmethod
    def load_card(self, name: str, env: Optional[str] = None) -> Dict[str, Any]:
        """Load the card of the specified asset.

        :param name:
            The name of the asset.
        :param env:
            The name of the environment within which to load the card.
        """

    @abstractmethod
    def save_card(
        self, name: str, card: Mapping[str, Any], env: Optional[str] = None
    ) -> None:
        """Save the card of the specified asset.

        :param name:
            The name of the asset.
        :param data:
            The card of the asset.
        :param:
            The name of the environment within which to save the card.
        """


@final
class LocalAssetCardStorage(AssetCardStorage):
    """Stores asset cards on a local file system."""

    base_pathname: Path

    def __init__(self, base_pathname: Path) -> None:
        """
        :param base_pathname:
            The pathname of the base directory under which to store asset cards.
        """
        self.base_pathname = base_pathname

    @finaloverride
    def load_card(self, name: str, env: Optional[str] = None) -> Dict[str, Any]:
        if os.sep in name or (os.altsep and os.altsep in name):
            raise ValueError(
                f"`name` must be a valid filename, but is '{name}' instead."
            )

        if env:
            filename = f"{name}@{env}"
        else:
            filename = name

        pathname = self.base_pathname.joinpath(filename).with_suffix(".yaml")

        try:
            fp = open(pathname)
        except FileNotFoundError as ex:
            raise AssetCardNotFoundError(
                f"An asset card with the name '{name}' cannot be found."
            ) from ex

        with fp:
            try:
                data = yaml.safe_load(fp)
            except YAMLError as ex:
                raise AssetCardError(
                    f"The asset card '{name}' cannot be loaded."
                ) from ex

            if not isinstance(data, dict):
                raise AssetCardError(
                    f"The data of the asset card '{name}' must be of type {dict}, but is of type {type(data)} instead."
                )

            return data

    @finaloverride
    def save_card(
        self, name: str, data: Mapping[str, Any], env: Optional[str] = None
    ) -> None:
        raise NotImplementedError()


class AssetCardNotFoundError(AssetError):
    """Raised when an asset card cannot be found."""
