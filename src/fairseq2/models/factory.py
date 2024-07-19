# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, Tuple, Type, TypeVar, final

from torch.nn import Module

from fairseq2.config_registry import ConfigRegistry
from fairseq2.typing import DataClass, DataType, Device
from fairseq2.utils.dataclass import FieldError, update_dataclass
from fairseq2.utils.value_converter import ValueConverter

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


class _GenericModelFactory(Protocol):
    def __call__(
        self,
        arch: Optional[str],
        config: Optional[Dict[str, Any]],
        device: Optional[Device],
        dtype: Optional[DataType],
    ) -> Tuple[Module, DataClass]:
        ...


@final
class DelegatingModelFactory:
    """Constructs models using registered factories."""

    _factories: Dict[str, _GenericModelFactory]

    def __init__(self) -> None:
        self._factories = {}

    def __call__(
        self,
        family: str,
        arch: Optional[str],
        config: Optional[Dict[str, Any]],
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
            The weakly-typed model configuration. Keys within ``config`` will
            override corresponding fields in model configuration object. Keys
            that are not present in ``config`` will have their default values
            set based on ``arch``.
        """

        try:
            factory = self._factories[family]
        except KeyError:
            raise ValueError("`family` must have a registered model factory.") from None

        return factory(arch, config, device, dtype)

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
            The model factory.
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

        def create_model(
            arch: Optional[str],
            config: Optional[Dict[str, Any]],
            device: Optional[Device],
            dtype: Optional[DataType],
        ) -> Tuple[Module, DataClass]:
            if arch is None:
                try:
                    config_ = config_kls()
                except TypeError as ex:
                    raise RuntimeError(
                        f"The {family} model family has not default configuration."
                    ) from ex
            else:
                if arch_configs is None:
                    raise ValueError(
                        f"`arch` must be a registered architecture, but the '{family}' model family has no architecture named '{arch}'."
                    )

                try:
                    config_ = arch_configs.get(arch)
                except ValueError:
                    raise ValueError(
                        f"`arch` must be a registered architecture, but the '{family}' model family has no architecture named '{arch}'."
                    ) from None

            if config is not None:
                try:
                    unknown_fields = update_dataclass(
                        config_, config, value_converter=value_converter
                    )
                except FieldError as ex:
                    raise ValueError(
                        f"`config` must be a valid model configuration, but the value of the configuration field '{ex.field_name}' is invalid. See nested exception for details."
                    ) from ex

                if unknown_fields:
                    raise ValueError(
                        f"`config` must be a valid model configuration, but the following configuration fields are unknown: {', '.join(unknown_fields)}"
                    )

            model = factory(config_, device=device, dtype=dtype)

            return model, config_

        self._factories[family] = create_model


create_model = DelegatingModelFactory()
