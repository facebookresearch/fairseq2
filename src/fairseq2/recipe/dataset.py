# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import final

from fairseq2.assets import AssetCardError, AssetStore
from fairseq2.datasets import (
    DatasetFamilyHandler,
    DatasetFamilyNotKnownError,
    DatasetNotKnownError,
)
from fairseq2.error import InternalError
from fairseq2.gang import GangError, Gangs, raise_operational_gang_error
from fairseq2.logging import log
from fairseq2.recipe.asset_config import AssetConfigOverrider
from fairseq2.recipe.config import DatasetSection
from fairseq2.runtime.provider import Provider


@final
class DatasetFactory:
    def __init__(
        self,
        section: DatasetSection,
        handlers: Provider[DatasetFamilyHandler],
        asset_store: AssetStore,
        asset_config_overrider: AssetConfigOverrider,
        gangs: Gangs,
    ) -> None:
        self._section = section
        self._handlers = handlers
        self._asset_store = asset_store
        self._asset_config_overrider = asset_config_overrider
        self._gangs = gangs

    def create(self) -> object:
        if self._section.name is not None:
            return self._open_dataset_from_card()

        if self._section.family is not None:
            return self._open_custom_dataset()

        raise InternalError("`section.name` and `section.family` are both `None`.")

    def _open_dataset_from_card(self) -> object:
        name = self._section.name
        if name is None:
            raise InternalError("`section.name` is `None`.")

        card = self._asset_store.maybe_retrieve_card(name)
        if card is None:
            raise DatasetNotKnownError(name)

        family = card.field("dataset_family").as_(str)

        handler = self._handlers.maybe_get(family)
        if handler is None:
            msg = f"family field of the {name} asset card is expected to be a supported dataset family, but is {family} instead."

            raise AssetCardError(name, msg)

        config = handler.get_dataset_config(card)

        if self._section.config_overrides is not None:
            config = self._asset_config_overrider.apply_overrides(
                "dataset", config, self._section.config_overrides
            )

        log.info("Opening {} dataset.", name)

        dataset = handler.open_dataset(card, config)

        try:
            self._gangs.root.barrier()
        except GangError as ex:
            raise_operational_gang_error(ex)

        log.info("Dataset loaded.")

        return dataset

    def _open_custom_dataset(self) -> object:
        family = self._section.family
        if family is None:
            raise InternalError("`section.family` is `None`.")

        handler = self._handlers.maybe_get(family)
        if handler is None:
            raise DatasetFamilyNotKnownError(family)

        try:
            config = handler.config_kls()
        except TypeError as ex:
            raise InternalError(
                f"Default configuration of the {family} dataset family cannot be constructed."
            ) from ex

        if self._section.config_overrides is not None:
            config = self._asset_config_overrider.apply_overrides(
                "tokenizer", config, self._section.config_overrides
            )

        log.info("Opening dataset.")

        dataset = handler.open_custom_dataset("custom", config)

        try:
            self._gangs.root.barrier()
        except GangError as ex:
            raise_operational_gang_error(ex)

        log.info("Dataset loaded.")

        return dataset
