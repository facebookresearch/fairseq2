# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from copy import deepcopy
from typing import Optional, Protocol, Type, TypeVar, Union, final

from fairseq2.assets import (
    AssetCard,
    AssetCardError,
    AssetCardFieldNotFoundError,
    AssetError,
    AssetStore,
)
from fairseq2.models.architecture_registry import ModelArchitectureRegistry
from fairseq2.utils.dataclass import update_dataclass

logger = logging.getLogger("fairseq2.models")


ModelConfigT = TypeVar("ModelConfigT")

ModelConfigT_co = TypeVar("ModelConfigT_co", covariant=True)


class ModelConfigLoader(Protocol[ModelConfigT_co]):
    """Loads model configurations of type ``ModelConfigT``."""

    def __call__(self, model_name_or_card: Union[str, AssetCard]) -> ModelConfigT_co:
        """
        :param model_name_or_card:
            The name or asset card of the model whose configuration to load.
        """


@final
class StandardModelConfigLoader(ModelConfigLoader[ModelConfigT]):
    """Loads model configurations of type ``ModelConfigT``."""

    _family: str
    _config_kls: Type[ModelConfigT]
    _archs: Optional[ModelArchitectureRegistry[ModelConfigT]]
    _asset_store: AssetStore

    def __init__(
        self,
        family: str,
        config_kls: Type[ModelConfigT],
        archs: Optional[ModelArchitectureRegistry[ModelConfigT]],
        asset_store: AssetStore,
    ) -> None:
        """
        :param family:
            The model family.
        :param config_kls:
            The type of the model configuration.
        :param asset_store:
            The asset store where to check for available models.
        :param archs:
            The registry containing all supported model architectures.
        """
        self._family = family
        self._config_kls = config_kls
        self._asset_store = asset_store
        self._archs = archs

    def __call__(self, model_name_or_card: Union[str, AssetCard]) -> ModelConfigT:
        if isinstance(model_name_or_card, AssetCard):
            card = model_name_or_card
        else:
            card = self._asset_store.retrieve_card(model_name_or_card)

        card.field("model_family").check_equals(self._family)

        # If the card holds a configuration object, it takes precedence.
        try:
            config = card.field("model_config").as_(self._config_kls)

            return deepcopy(config)
        except AssetCardError:
            pass

        arch = None

        if self._archs is not None:
            try:
                # Ensure that the card has a valid model architecture.
                arch = card.field("model_arch").as_one_of(self._archs.names())
            except AssetCardFieldNotFoundError:
                pass

        # Load the model configuration.
        if arch is None:
            try:
                config = self._config_kls()
            except TypeError as ex:
                raise AssetError(
                    f"The {self._family} model family has no default configuration."
                ) from ex
        else:
            assert self._archs is not None

            try:
                config = self._archs.get_config(arch)
            except ValueError as ex:
                raise AssetError(
                    f"The {self._family} model family has no architecture named '{arch}'."
                ) from ex

        # Check if we should override anything in the default model
        # configuration.
        try:
            config_overrides = card.field("model_config").as_(dict)
        except AssetCardFieldNotFoundError:
            config_overrides = None

        if config_overrides:
            try:
                update_dataclass(config, deepcopy(config_overrides))
            except (TypeError, ValueError) as ex:
                raise AssetError(
                    f"The value of the field `model_config` of the asset card '{card.name}' contains one or more invalid keys. See nested exception for details."
                ) from ex

        return config
