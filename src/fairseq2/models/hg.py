# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
This module provides an API for converting state dicts and configurations of
fairseq2 models to their Hugging Face Transformer equivalents.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, cast, final

import huggingface_hub
import transformers
from torch import Tensor
from transformers import PretrainedConfig
from typing_extensions import override

from fairseq2.error import NotSupportedError
from fairseq2.models.family import HuggingFaceExporter
from fairseq2.runtime.dependency import get_dependency_resolver


@dataclass
class HuggingFaceConfig:
    """
    Represents the configuration of a Hugging Face Transformers model.

    This class is part of the :class:`HuggingFaceConverter` interface which
    converts fairseq2 models to their Hugging Face equivalents.
    """

    data: Mapping[str, object]
    """
    Configuration data.

    Each key in this mapping must correspond to an attribute of the actual
    configuration class in Hugging Face Transformers.
    """

    kls_name: str
    """
    Name of the configuration class in Hugging Face Transformers. For instance,
    Qwen3Config or LlamaConfig.
    """

    arch: str | Sequence[str]
    """
    Architecture(s) of the model as defined in Hugging Face Transformers. For
    instance, Qwen3ForCausalLM, LlamaForCausalLM.
    """


class HuggingFaceConverter(ABC):
    """
    Converts the state dict and configuration of a fairseq2 model to its Hugging
    Face Transformers equivalent.

    Model authors must register their converter implementations with fairseq2
    as part of library initialization as shown below:

    .. code:: python

        from fairseq2.models.hg import HuggingFaceConverter
        from fairseq2.runtime.dependency import DependencyContainer, register_model_family

        class MyModelConverter(HuggingFaceConverter):
            ...

        def register_my_model(container: DependencyContainer) -> None:
            register_model_family(container, name="my_model_family", ...)

            container.register_type(
                HuggingFaceConverter, MyModelConverter, key="my_model_family",
            )
    """

    @abstractmethod
    def to_hg_config(self, config: object) -> HuggingFaceConfig:
        """
        Converts the specified fairseq2 model configuration to its Hugging Face
        Transformers equivalent.

        :raises TypeError: ``config`` is not of valid type. The expected type
            is one registered as part of the :class:`ModelFamily`.
        """

    @abstractmethod
    def to_hg_state_dict(
        self, state_dict: dict[str, object], config: object
    ) -> dict[str, object]:
        """
        Converts the specified fairseq2 state dict to its Hugging Face
        Transformers equivalent.

        ``config`` is the fairseq2 model configuration and can be used to
        adjust the converted state dict when necessary.

        :raises TypeError: ``config`` is not of valid type. The expected type
            is one registered as part of the :class:`ModelFamily`.
        """


# TODO: Remove in v0.9
@final
class _LegacyHuggingFaceConverter(HuggingFaceConverter):
    def __init__(self, exporter: HuggingFaceExporter[Any]) -> None:
        self._exporter = exporter

    @override
    def to_hg_config(self, config: object) -> HuggingFaceConfig:
        raise NotSupportedError()

    @override
    def to_hg_state_dict(
        self, state_dict: dict[str, object], config: object
    ) -> dict[str, object]:
        raise NotSupportedError()


def get_hugging_face_converter(family_name: str) -> HuggingFaceConverter:
    """
    Returns the :class:`HuggingFaceConverter` of the specified model family.

    :raises NotSupportedError: The model family does not support Hugging Face
        conversion.
    """
    resolver = get_dependency_resolver()

    hg_converter = resolver.resolve_optional(HuggingFaceConverter, key=family_name)
    if hg_converter is None:
        raise NotSupportedError(
            f"{family_name} model family does not support Hugging Face conversion."
        )

    return hg_converter


def save_hugging_face_model(
    save_dir: Path, state_dict: dict[str, object], config: HuggingFaceConfig
) -> None:
    """
    Saves the state dict and configuration of a Hugging Face Transformers model
    to the specified directory.

    :raises TypeError: ``config.kls_name`` does not correspond to the expected
        :class:`PretrainedConfig` subclass of the Hugging Face model.

    :raises TypeError: ``state_dict`` contains non-tensor values which is not
        supported in Safetensors format.

    :raises ValueError: A key in ``config`` does not have a corresponding
        attribute in Hugging Face model configuration class.

    :raises OSError: The state dict or configuration cannot be saved to the
        file system.
    """
    try:
        config_kls = getattr(transformers, config.kls_name)
    except AttributeError:
        raise TypeError(f"`transformers.{config.kls_name}` is not a type.") from None

    if not issubclass(config_kls, PretrainedConfig):
        raise TypeError(
            f"`transformers.{config.kls_name}` is expected to be a subclass of `{PretrainedConfig}`."
        )

    native_config = config_kls()

    for key, value in config.data.items():
        if not hasattr(native_config, key):
            raise ValueError(
                f"`transformers.{config.kls_name}` does not have an attribute named {key}."
            )

        setattr(native_config, key, value)

    arch = config.arch

    setattr(native_config, "architectures", [arch] if isinstance(arch, str) else arch)

    native_config.save_pretrained(save_dir)

    for key, value in state_dict.items():
        if not isinstance(value, Tensor):
            raise TypeError(
                f"`state_dict[{key}]` must be of type `{Tensor}`, but is of type `{type(value)}` instead."
            )

    tensors = cast(dict[str, Tensor], state_dict)

    huggingface_hub.save_torch_state_dict(tensors, save_dir)
