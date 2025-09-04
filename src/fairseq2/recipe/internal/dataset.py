# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import final

from fairseq2.assets import AssetStore
from fairseq2.datasets import (
    DatasetFamily,
    DatasetFamilyNotKnownError,
    DatasetNotKnownError,
    get_dataset_family,
)
from fairseq2.error import InternalError
from fairseq2.gang import GangError, Gangs, raise_operational_gang_error
from fairseq2.logging import log
from fairseq2.recipe.config import DatasetSection
from fairseq2.recipe.dataset import RecipeDataset
from fairseq2.recipe.error import ErrorContext
from fairseq2.recipe.internal.asset_config import _AssetConfigOverrider
from fairseq2.recipe.internal.log import _LogHelper
from fairseq2.runtime.lookup import Lookup


@final
class _RecipeDatasetOpener:
    def __init__(
        self,
        families: Lookup[DatasetFamily],
        asset_store: AssetStore,
        asset_config_overrider: _AssetConfigOverrider,
        gangs: Gangs,
        log_helper: _LogHelper,
    ) -> None:
        self._families = families
        self._asset_store = asset_store
        self._asset_config_overrider = asset_config_overrider
        self._gangs = gangs
        self._log_helper = log_helper

    def open(self, section_name: str, section: DatasetSection) -> RecipeDataset:
        try:
            if section.name is not None:
                if section.family is not None:
                    log.warning("`{0}.family` will be ignored since `{0}.name` is specified.", section_name)  # fmt: skip

                return self._open_dataset(section_name, section)

            if section.family is not None:
                return self._open_custom_dataset(section_name, section)
        except Exception as ex:
            ErrorContext.set_config_section_name(ex, section_name)

            raise

        raise InternalError("`section.name` and `section.family` are both `None`.")

    def _open_dataset(
        self, section_name: str, section: DatasetSection
    ) -> RecipeDataset:
        name = section.name
        if name is None:
            raise InternalError("`section.name` is `None`.")

        card = self._asset_store.maybe_retrieve_card(name)
        if card is None:
            raise DatasetNotKnownError(name)

        family = get_dataset_family(card, self._families)

        config = family.get_dataset_config(card)

        if section.config_overrides is not None:
            config = self._asset_config_overrider.apply_overrides(
                section_name, config, section.config_overrides
            )

        if section_name == "dataset":
            log.info("Opening {} dataset.", name)
        else:
            log.info("Opening {} dataset specified in `{}` section.", name, section_name)  # fmt: skip

        if config is not None:
            self._log_helper.log_config("Dataset Config", config)

        inner_dataset = family.open_dataset(card, config)

        try:
            self._gangs.root.barrier()
        except GangError as ex:
            raise_operational_gang_error(ex)

        log.info("Dataset loaded.")

        return RecipeDataset(inner_dataset, config, family)

    def _open_custom_dataset(
        self, section_name: str, section: DatasetSection
    ) -> RecipeDataset:
        family_name = section.family
        if family_name is None:
            raise InternalError("`section.family` is `None`.")

        family = self._families.maybe_get(family_name)
        if family is None:
            raise DatasetFamilyNotKnownError(family_name)

        try:
            config = family.config_kls()
        except TypeError as ex:
            raise InternalError(
                f"Default configuration of the {family_name} dataset family cannot be constructed."
            ) from ex

        if section.config_overrides is not None:
            config = self._asset_config_overrider.apply_overrides(
                section_name, config, section.config_overrides
            )

        if section_name == "dataset":
            log.info("Opening dataset.")
        else:
            log.info("Opening dataset specified in `{} section.", section_name)

        if config is not None:
            self._log_helper.log_config("Dataset Config", config)

        inner_dataset = family.open_custom_dataset(config)

        try:
            self._gangs.root.barrier()
        except GangError as ex:
            raise_operational_gang_error(ex)

        log.info("Dataset loaded.")

        return RecipeDataset(inner_dataset, config, family)
