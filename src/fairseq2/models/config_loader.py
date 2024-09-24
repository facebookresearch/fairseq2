# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Protocol, TypeVar, cast, final

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
from fairseq2.utils.structured import (
    StructuredError,
    ValueConverter,
    default_value_converter,
    merge_unstructured,
)

ModelConfigT = TypeVar("ModelConfigT", bound=DataClass)

ModelConfigT_co = TypeVar("ModelConfigT_co", bound=DataClass, covariant=True)


class ModelConfigLoader(Protocol[ModelConfigT_co]):
    """Loads model configurations of type ``ModelConfigT``."""

    def __call__(self, model_name_or_card: str | AssetCard) -> ModelConfigT_co:
        """
        :param model_name_or_card:
            The name or the asset card of the model whole configuration to load.
        """


@final
class StandardModelConfigLoader(ModelConfigLoader[ModelConfigT]):
    """Loads model configurations of type ``ModelConfigT``."""

    _asset_store: AssetStore
    _family: str
    _config_kls: type[ModelConfigT]
    _arch_configs: ConfigRegistry[ModelConfigT] | None

    def __init__(
        self,
        family: str,
        config_kls: type[ModelConfigT],
        arch_configs: ConfigRegistry[ModelConfigT] | None,
        *,
        asset_store: AssetStore | None = None,
        value_converter: ValueConverter | None = None,
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

    def __call__(self, model_name_or_card: str | AssetCard) -> ModelConfigT:
        if isinstance(model_name_or_card, AssetCard):
            card = model_name_or_card
        else:
            card = self._asset_store.retrieve_card(model_name_or_card)

        model_family = get_model_family(card)
        if model_family != self._family:
            raise AssetCardError(
                f"The value of the field 'model_family' of the asset card '{card.name}' must be '{self._family}', but is '{model_family}' instead."
            )

        config_kls = self._config_kls

        try:
            arch = card.field("model_arch").as_(str)
        except AssetCardFieldNotFoundError:
            arch = None

        # Load the configuration.
        if arch is None:
            try:
                config = config_kls()
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

        # Override the default architecture configuration if needed.
        config_overrides_list = []

        card_: AssetCard | None = card

        while card_ is not None:
            if "model_config" in card_.metadata:
                config_overrides = card_.field("model_config").as_unstructured()

                config_overrides_list.append(config_overrides)

            card_ = card_.base

        if config_overrides_list:
            try:
                unstructured_config = self._value_converter.unstructure(config)
            except StructuredError as ex:
                raise AssetError(
                    f"The model configuration class of the '{self._family}' cannot be used. Please file a bug report to the model author."
                ) from ex

            try:
                for config_overrides in reversed(config_overrides_list):
                    unstructured_config = merge_unstructured(
                        unstructured_config, config_overrides
                    )

                config = self._value_converter.structure(
                    unstructured_config, type_expr=config_kls
                )
            except StructuredError as ex:
                raise AssetError(
                    f"The value of the field 'model_config' of the asset card '{card.name}' cannot be parsed as a valid model configuration. Please file a bug report to the asset author."
                ) from ex

        return config


def is_model_card(card: AssetCard) -> bool:
    """Return ``True`` if ``card`` specifies a model."""
    return card.field("model_family").exists() or card.field("model_type").exists()


def get_model_family(card: AssetCard) -> str:
    """Return the model family name contained in ``card``."""
    try:
        return cast(str, card.field("model_family").as_(str))
    except AssetCardFieldNotFoundError:
        pass

    try:
        # Compatibility with older fairseq2 versions.
        return cast(str, card.field("model_type").as_(str))
    except AssetCardFieldNotFoundError:
        raise AssetCardFieldNotFoundError(
            f"The asset card '{card.name}' must have a field named 'model_family."
        ) from None
