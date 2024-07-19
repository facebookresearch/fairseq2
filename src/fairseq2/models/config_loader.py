# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, Type, TypeVar, Union, final

from fairseq2.assets import (
    AssetCard,
    AssetCardError,
    AssetCardFieldNotFoundError,
    AssetError,
    AssetStore,
    default_asset_store,
)
from fairseq2.config_registry import ConfigRegistry
from fairseq2.typing import DataClass
from fairseq2.utils.dataclass import FieldError, update_dataclass
from fairseq2.utils.value_converter import ValueConverter, default_value_converter

ModelConfigT = TypeVar("ModelConfigT", bound=DataClass)

ModelConfigT_co = TypeVar("ModelConfigT_co", bound=DataClass, covariant=True)


class ModelConfigLoader(Protocol[ModelConfigT_co]):
    """Loads model configurations of type ``ModelConfigT``."""

    def __call__(self, model_name_or_card: Union[str, AssetCard]) -> ModelConfigT_co:
        """
        :param model_name_or_card:
            The name or the asset card of the model whole configuration to load.
        """


@final
class StandardModelConfigLoader(ModelConfigLoader[ModelConfigT]):
    """Loads model configurations of type ``ModelConfigT``."""

    _asset_store: AssetStore
    _family: str
    _config_kls: Type[ModelConfigT]
    _arch_configs: Optional[ConfigRegistry[ModelConfigT]]
    _value_converter: ValueConverter

    def __init__(
        self,
        *,
        family: str,
        config_kls: Type[ModelConfigT],
        arch_configs: Optional[ConfigRegistry[ModelConfigT]],
        asset_store: Optional[AssetStore] = None,
        value_converter: Optional[ValueConverter] = None,
    ) -> None:
        """
        :param family:
            The model family.
        :param config_kls:
            The type of the model configuration.
        :param arch_configs:
            The registry containing all supported model architectures.
        :param asset_store:
            The asset store where to check for available models. If ``None``,
            the default asset store will be used.
        :param value_converter:
            The :class:`ValueConverter` instance to use. If ``None``, the
            default instance will be used.
        """
        self._asset_store = asset_store or default_asset_store
        self._family = family
        self._config_kls = config_kls
        self._arch_configs = arch_configs
        self._value_converter = value_converter or default_value_converter

    def __call__(self, model_name_or_card: Union[str, AssetCard]) -> ModelConfigT:
        if isinstance(model_name_or_card, AssetCard):
            card = model_name_or_card
        else:
            card = self._asset_store.retrieve_card(model_name_or_card)

        model_family = get_model_family(card)
        if model_family != self._family:
            raise AssetCardError(
                f"The value of the field 'model_family' of the asset card '{card.name}' must be '{self._family}', but is '{model_family}' instead."
            )

        try:
            arch = card.field("model_arch").as_(str)
        except AssetCardFieldNotFoundError:
            arch = None

        # Load the configuration.
        if arch is None:
            try:
                config = self._config_kls()
            except TypeError as ex:
                raise AssetError(
                    f"The '{self._family}' model family has no default configuration."
                ) from ex
        else:
            if self._arch_configs is None:
                raise AssetError(
                    f"The '{self._family}' model family has no architecture named '{arch}'."
                )

            try:
                config = self._arch_configs.get(arch)
            except ValueError:
                raise AssetError(
                    f"The '{self._family}' model family has no architecture named '{arch}'."
                ) from None

        # Check whether to override anything in the default configuration.
        if config_overrides := card.field("model_config").get_as_(Dict[str, Any]):
            try:
                unknown_fields = update_dataclass(
                    config, config_overrides, value_converter=self._value_converter
                )
            except FieldError as ex:
                raise AssetCardError(
                    f"The value of the field 'model_config' of the asset card '{card.name}' must be a valid model configuration, but the value of the configuration field '{ex.field_name}' is invalid. See nested exception for details."
                ) from ex

            if unknown_fields:
                raise AssetCardError(
                    f"The value of the field 'model_config' of the asset card '{card.name}' must be a valid model configuration, but the following configuration fields are unknown: {', '.join(unknown_fields)}"
                )

        return config


def is_model_card(card: AssetCard) -> bool:
    """Return ``True`` if ``card`` specifies a model."""
    return card.field("model_family").exists() or card.field("model_type").exists()


def get_model_family(card: AssetCard) -> str:
    """Return the model family name contained in ``card``."""
    try:
        return card.field("model_family").as_(str)  # type: ignore[no-any-return]
    except AssetCardFieldNotFoundError:
        pass

    try:
        # Compatibility with older fairseq2 versions.
        return card.field("model_type").as_(str)  # type: ignore[no-any-return]
    except AssetCardFieldNotFoundError:
        raise AssetCardFieldNotFoundError(
            f"The asset card '{card.name}' must have a field named 'model_family."
        ) from None
