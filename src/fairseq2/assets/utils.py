# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import Optional, Union

from fairseq2.assets.card import AssetCard
from fairseq2.assets.metadata_provider import (
    AssetMetadataError,
    AssetNotFoundError,
    InProcAssetMetadataProvider,
    _load_metadata_file,
)
from fairseq2.assets.store import AssetStore, StandardAssetStore, default_asset_store


def retrieve_asset_card(
    name_or_card: Union[str, AssetCard, Path], store: Optional[AssetStore] = None
) -> AssetCard:
    """Retrieve the specified asset.

    :param name_or_card:
        The name, card, or path to the card file of the asset to load.
    :param store:
        The asset store where to check for available assets. If ``None``, the
        default asset store will be used.
    """
    if isinstance(name_or_card, AssetCard):
        return name_or_card

    if store is None:
        store = default_asset_store

    if isinstance(name_or_card, Path):
        return _card_from_file(name_or_card, store)

    name = name_or_card

    try:
        return store.retrieve_card(name)
    except AssetNotFoundError:
        pass

    try:
        file = Path(name)
    except ValueError:
        file = None

    if file is not None:
        if (file.suffix == ".yaml" or file.suffix == ".yml") and file.exists():
            return _card_from_file(file, store)

    raise AssetNotFoundError(name, f"An asset with the name '{name}' cannot be found.")


def _card_from_file(file: Path, store: AssetStore) -> AssetCard:
    if not isinstance(store, StandardAssetStore):
        raise ValueError(
            f"`store` must be of type `{StandardAssetStore}` when `name_or_card` is a pathname, but is of type {type(store)} instead."
        )

    all_metadata = _load_metadata_file(file)

    if len(all_metadata) != 1:
        raise AssetMetadataError(
            f"The specified asset metadata file '{file}' contains metadata for more than one asset."
        )

    name, metadata = all_metadata[0]

    metadata["name"] = name

    metadata_provider = InProcAssetMetadataProvider([metadata], name="argument")

    # Strip the environment tag.
    name, _ = name.split("@", maxsplit=1)

    return store.retrieve_card(name, extra_provider=metadata_provider)
