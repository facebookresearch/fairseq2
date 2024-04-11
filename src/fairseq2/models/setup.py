# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple, Type, TypeVar

from torch.nn import Module

from fairseq2.assets import default_asset_store, default_download_manager
from fairseq2.models.architecture_registry import ModelArchitectureRegistry
from fairseq2.models.config_loader import ModelConfigLoader, StandardModelConfigLoader
from fairseq2.models.loader import (
    CheckpointConverter,
    ModelFactory,
    ModelLoader,
    StandardModelLoader,
    load_model,
)

ModelT = TypeVar("ModelT", bound=Module)

ModelConfigT = TypeVar("ModelConfigT")


def setup_model(
    name: str,
    config_kls: Type[ModelConfigT],
    factory: ModelFactory[ModelConfigT, ModelT],
    archs: Optional[ModelArchitectureRegistry[ModelConfigT]] = None,
    checkpoint_converter: Optional[CheckpointConverter[ModelConfigT]] = None,
    *,
    mmap: bool = False,
    restrict_checkpoints: bool = True,
    skip_meta_init: bool = False,
) -> Tuple[ModelLoader[ModelT], ModelConfigLoader[ModelConfigT]]:
    """Set up a model.

    :param name:
        The name of the model family.
    :param config_kls:
        The type of the model configuration.
    :param factory:
        The factory to construct models.
    :param archs:
        The registry containing all supported model architectures.
    :param checkpoint_converter:
        The converter to which loaded checkpoints will be passed for further
        processing.
    :param mmap:
        If ``True``, indicates whether checkpoints should be memory mapped.
    :param restrict_checkpoints:
        If ``True``, restricts the Python unpickler to load only tensors,
        primitive types, and dictionaries while loading checkpoints.
    :param skip_meta_init:
        If ``True``, skips meta device initialization and constructs models
        directly on the requested device. Meant to be used with models that do
        not support PyTorch's ``reset_parameters()`` convention.

    :returns:
        - The model loader.
        - The model configuration loader.
    """
    config_loader = StandardModelConfigLoader(
        name, config_kls, archs, default_asset_store
    )

    loader = StandardModelLoader(
        default_asset_store,
        default_download_manager,
        config_loader,
        factory,
        checkpoint_converter,
        mmap=mmap,
        restrict_checkpoints=restrict_checkpoints,
        skip_meta_init=skip_meta_init,
    )

    load_model.register_loader(name, loader)

    return loader, config_loader
