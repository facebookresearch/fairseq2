# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

from fairseq2.assets import (
    AssetCard,
    AssetMetadataError,
    AssetNotFoundError,
    InProcAssetMetadataProvider,
    default_asset_store,
    load_metadata_file,
)

AssetReference: TypeAlias = str | AssetCard | Path


def retrieve_asset_card(name_or_card: AssetReference) -> AssetCard:
    """Retrieve the specified asset.

    :param name_or_card:
        The name, card, or path to the card file of the asset to load.
    """
    if isinstance(name_or_card, AssetCard):
        return name_or_card

    if isinstance(name_or_card, Path):
        if name_or_card.is_dir():
            raise AssetNotFoundError(
                f"{name_or_card}", f"An asset metadata file cannot be found at {name_or_card}."  # fmt: skip
            )

        return _card_from_file(name_or_card)

    name = name_or_card

    try:
        return default_asset_store.retrieve_card(name)
    except AssetNotFoundError:
        pass

    try:
        file = Path(name)
    except ValueError:
        file = None

    if file is not None:
        if (file.suffix == ".yaml" or file.suffix == ".yml") and file.exists():
            return _card_from_file(file)

    raise AssetNotFoundError(name, f"An asset with the name '{name}' cannot be found.")


def _card_from_file(file: Path) -> AssetCard:
    all_metadata = load_metadata_file(file)

    if len(all_metadata) != 1:
        raise AssetMetadataError(
            f"The specified asset metadata file '{file}' contains metadata for more than one asset."
        )

    name, metadata = all_metadata[0]

    metadata["name"] = name

    metadata_provider = InProcAssetMetadataProvider([metadata], name="argument")

    # Strip the environment tag.
    name, _ = name.split("@", maxsplit=1)

    return default_asset_store.retrieve_card(name, extra_provider=metadata_provider)


def asset_as_path(name_or_card: AssetReference) -> Path:
    if isinstance(name_or_card, Path):
        return name_or_card

    if isinstance(name_or_card, AssetCard):
        raise ValueError(
            f"`name_or_card` must be of type `{str}` or `{Path}`, but is of type `{AssetCard}` instead."
        )

    try:
        path = Path(name_or_card)
    except ValueError:
        path = None

    if path is None or not path.exists():
        raise AssetNotFoundError(
            name_or_card, f"An asset with the name '{name_or_card}' cannot be found."
        ) from None

    return path
