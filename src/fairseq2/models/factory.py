# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import (
    Any,
    Dict,
    Generic,
    Mapping,
    Optional,
    Protocol,
    Tuple,
    Type,
    TypeVar,
    final,
)

from torch.nn import Module

from fairseq2.config_registry import ConfigRegistry
from fairseq2.typing import DataClass, DataType, Device
from fairseq2.utils.dataclass import FieldError, update_dataclass
from fairseq2.utils.value_converter import ValueConverter, default_value_converter

ModelT = TypeVar("ModelT", bound=Module)

ModelT_co = TypeVar("ModelT_co", bound=Module, covariant=True)

ModelConfigT = TypeVar("ModelConfigT", bound=DataClass)

ModelConfigT_contra = TypeVar(
    "ModelConfigT_contra", bound=DataClass, contravariant=True
)


class ModelFactory(Protocol[ModelConfigT_contra, ModelT_co]):
    """Constructs models of type ``ModelT``."""

    def __call__(
        self,
        config: ModelConfigT_contra,
        *,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> ModelT_co:
        """
        :param config:
            The model configuration.
        :param device:
            The device on which to initialize the model.
        :param dtype:
            The data type of the model parameters and buffers.
        """


class GenericModelFactory(Protocol):
    """Constructs models."""

    def __call__(
        self,
        family: str,
        arch: Optional[str],
        config: Any,
        *,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> Tuple[Module, DataClass]:
        """
        :param family:
            The family of the model.
        :param arch:
            The architecture of the model. If ``None``, uses the default model
            configuration.
        :param config:
            The model configuration object or a dictionary where keys will
            override corresponding fields in the model configuration of ``arch``.
        """


@final
class StandardGenericModelFactory(GenericModelFactory, Generic[ModelT, ModelConfigT]):
    """Constructs models."""

    _family: str
    _factory: ModelFactory[ModelConfigT, ModelT]
    _config_kls: Type[ModelConfigT]
    _arch_configs: Optional[ConfigRegistry[ModelConfigT]]
    _value_converter: ValueConverter

    def __init__(
        self,
        *,
        family: str,
        factory: ModelFactory[ModelConfigT, ModelT],
        config_kls: Type[ModelConfigT],
        arch_configs: Optional[ConfigRegistry[ModelConfigT]],
        value_converter: Optional[ValueConverter] = None,
    ) -> None:
        """
        :param family:
            The model family.
        :param factory:
            The factory to construct models.
        :param config_kls:
            The type of the model configuration.
        :param arch_configs:
            The registry containing all supported model architectures.
        :param value_converter:
            The :class:`ValueConverter` instance to use. If ``None``, the
            default instance will be used.
        """
        self._family = family
        self._factory = factory
        self._config_kls = config_kls
        self._arch_configs = arch_configs
        self._value_converter = value_converter or default_value_converter

    def __call__(
        self,
        family: str,
        arch: Optional[str],
        config: Any,
        *,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> Tuple[Module, DataClass]:
        if family != self._family:
            raise ValueError(
                f"`family` must be '{self._family}', but is '{family}' instead."
            )

        if isinstance(config, self._config_kls):
            model = self._factory(config, device=device, dtype=dtype)

            return model, config

        if arch is None:
            try:
                config_ = self._config_kls()
            except TypeError as ex:
                raise RuntimeError(
                    f"The '{family}' model family has not default configuration."
                ) from ex
        else:
            if self._arch_configs is None:
                raise ValueError(
                    f"`arch` must be a registered architecture, but the '{family}' model family has no architecture named '{arch}'."
                )

            try:
                config_ = self._arch_configs.get(arch)
            except ValueError:
                raise ValueError(
                    f"`arch` must be a registered architecture, but the '{family}' model family has no architecture named '{arch}'."
                ) from None

        if config is not None:
            if not isinstance(config, Mapping):
                raise ValueError(
                    f"`config` must be of type `{self._config_kls}` or `{Mapping}`, but is of type `{type(config)}` instead."
                )

            try:
                unknown_fields = update_dataclass(
                    config_, config, value_converter=self._value_converter
                )
            except FieldError as ex:
                raise ValueError(
                    f"`config` must be a valid model configuration, but the value of the configuration field '{ex.field_name}' is invalid. See nested exception for details."
                ) from ex

            if unknown_fields:
                raise ValueError(
                    f"`config` must be a valid model configuration, but the following configuration fields are unknown: {', '.join(unknown_fields)}"
                )

        model = self._factory(config_, device=device, dtype=dtype)

        return model, config_


@final
class DelegatingGenericModelFactory(GenericModelFactory):
    """Constructs models using registered factories."""

    _factories: Dict[str, GenericModelFactory]

    def __init__(self) -> None:
        self._factories = {}

    def __call__(
        self,
        family: str,
        arch: Optional[str],
        config: Any,
        *,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> Tuple[Module, DataClass]:
        try:
            factory = self._factories[family]
        except KeyError:
            raise ValueError(
                f"`family` must be a supported model family, but '{family}' has no registered factory."
            ) from None

        return factory(family, arch, config, device=device, dtype=dtype)

    def register(
        self,
        *,
        family: str,
        factory: ModelFactory[ModelConfigT, ModelT],
        config_kls: Type[ModelConfigT],
        arch_configs: Optional[ConfigRegistry[ModelConfigT]],
        value_converter: Optional[ValueConverter] = None,
    ) -> None:
        """Register a model factory.

        :param family:
            The model family supported by ``factory``.
        :param factory:
            The factory to construct models.
        :param config_kls:
            The type of the model configuration.
        :param arch_configs:
            The registry containing all supported model architectures.
        :param value_converter:
            The :class:`ValueConverter` instance to use. If ``None``, the
            default instance will be used.
        """
        if family in self._factories:
            raise ValueError(
                f"`family` must be a unique model family name, but '{family}' has already a registered factory."
            )

        generic_factory = StandardGenericModelFactory(
            family=family,
            factory=factory,
            config_kls=config_kls,
            arch_configs=arch_configs,
            value_converter=value_converter,
        )

        self._factories[family] = generic_factory


create_model = DelegatingGenericModelFactory()
