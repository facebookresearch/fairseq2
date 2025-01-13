# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import TypeVar

from fairseq2.assets import AssetCard
from fairseq2.context import get_runtime_context
from fairseq2.datasets.handler import (
    DatasetHandler,
    DatasetNotFoundError,
    get_dataset_family,
)
from fairseq2.error import ContractError

DatasetT = TypeVar("DatasetT")


def load_dataset(
    name_or_card: str | AssetCard, kls: type[DatasetT], *, force: bool = False
) -> DatasetT:
    context = get_runtime_context()

    if isinstance(name_or_card, AssetCard):
        card = name_or_card
    else:
        card = context.asset_store.retrieve_card(name_or_card)

    family = get_dataset_family(card)

    registry = context.get_registry(DatasetHandler)

    try:
        handler = registry.get(family)
    except LookupError:
        raise DatasetNotFoundError(card.name) from None

    if not issubclass(handler.kls, kls):
        raise TypeError(
            f"The dataset is expected to be of type `{kls}`, but is of type `{type(handler.kls)}` instead."
        )

    dataset = handler.load(card, force=force)

    if not isinstance(dataset, kls):
        raise ContractError(
            f"The dataset is expected to be of type `{kls}`, but is of type `{type(handler.kls)}` instead."
        )

    return dataset
