# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from contextlib import nullcontext
from typing import cast, final

import torch
from torch import Tensor
from torch.nn import Module
from typing_extensions import override

from fairseq2.assets import (
    AssetCardError,
    AssetCardFieldNotFoundError,
    AssetCardNotFoundError,
    AssetMetadataSaveError,
    AssetStore,
)
from fairseq2.checkpoint import CheckpointAssetMetadataSaver
from fairseq2.dependency import DependencyResolver
from fairseq2.error import ContractError, NotSupportedError, SetupError
from fairseq2.file_system import FileSystem
from fairseq2.gang import GangError, Gangs
from fairseq2.logging import log
from fairseq2.models import (
    ModelConfigLoadError,
    ModelHandler,
    ModelLoadError,
    UnknownModelArchitectureError,
    UnknownModelError,
    UnknownModelFamilyError,
    model_asset_card_error,
)
from fairseq2.nn.utils.grad import clip_grad_norm
from fairseq2.recipe.checkpoint import check_has_checkpoint
from fairseq2.recipe.compile import compile_model
from fairseq2.recipe.config import (
    ModelSection,
    TrainerSection,
    get_config_section,
    get_output_dir,
)
from fairseq2.recipe.error import (
    ActivationCheckpointingNotSupportedError,
    ModelParallelismNotSupportedError,
    ModelPathNotFoundError,
)
from fairseq2.recipe.model import Model
from fairseq2.recipe.utils.log import log_config
from fairseq2.typing import ContextManager, DataClass, Provider, is_dataclass_instance
from fairseq2.utils.merge import MergeError, merge_dataclass
from fairseq2.utils.structured import StructureError, structure
from fairseq2.utils.yaml import YamlDumper


def load_base_model(resolver: DependencyResolver) -> Model:
    model_section = get_config_section(resolver, "model", ModelSection)

    trainer_section = get_config_section(resolver, "trainer", TrainerSection)

    handlers = resolver.resolve_provider(ModelHandler)

    gangs = resolver.resolve(Gangs)

    has_checkpoint = check_has_checkpoint(resolver)

    loader: _ModelLoader

    if model_section.path is not None:
        loader = _PathBasedModelLoader(handlers, has_checkpoint)
    elif model_section.name is not None:
        asset_store = resolver.resolve(AssetStore)

        loader = _CardBasedModelLoader(asset_store, handlers, has_checkpoint)
    elif model_section.family is not None:
        loader = _NewModelLoader(handlers, has_checkpoint)
    else:
        raise ValueError(
            "Either `model_section.name` or `model_section.family` must be specified."
        )

    try:
        model = loader.load(resolver, model_section, trainer_section, gangs)
    except ModelLoadError as ex:
        raise SetupError(
            f"The '{ex.model_name}' model cannot be loaded. See the nested exception for details."
        ) from ex
    except AssetMetadataSaveError as ex:
        raise SetupError(
            "The model card cannot be saved to the checkpoint directory. See the nested exception for details."
        ) from ex

    _save_model_metadata(resolver, model)

    return model


class _ModelLoader(ABC):
    @abstractmethod
    def load(
        self,
        resolver: DependencyResolver,
        model_section: ModelSection,
        trainer_section: TrainerSection,
        gangs: Gangs,
    ) -> Model: ...


@final
class _CardBasedModelLoader(_ModelLoader):
    _asset_store: AssetStore
    _handlers: Provider[ModelHandler]
    _has_checkpoint: bool

    def __init__(
        self,
        asset_store: AssetStore,
        handlers: Provider[ModelHandler],
        has_checkpoint: bool,
    ) -> None:
        self._asset_store = asset_store
        self._handlers = handlers
        self._has_checkpoint = has_checkpoint

    @override
    def load(
        self,
        resolver: DependencyResolver,
        model_section: ModelSection,
        trainer_section: TrainerSection,
        gangs: Gangs,
    ) -> Model:
        name = model_section.name
        if name is None:
            raise ValueError("`model_section.name` must be specified.")

        try:
            card = self._asset_store.retrieve_card(name)
        except AssetCardNotFoundError:
            raise UnknownModelError(name) from None
        except AssetCardError as ex:
            raise model_asset_card_error(name) from ex

        try:
            family = card.field("model_family").as_(str)
        except AssetCardFieldNotFoundError:
            raise UnknownModelError(name) from None
        except AssetCardError as ex:
            raise model_asset_card_error(name) from ex

        try:
            handler = self._handlers.resolve(family)
        except LookupError:
            raise UnknownModelFamilyError(family, name) from None

        if gangs.root.size != gangs.dp.size:
            if not handler.supports_model_parallelism:
                raise ModelParallelismNotSupportedError(name)

        try:
            config = handler.load_config(card)
        except ModelConfigLoadError as ex:
            raise ModelLoadError(
                name, f"The '{name}' model configuration cannot be loaded. See the nested exception for details."  # fmt: skip
            ) from ex

        config = _apply_config_overrides(config, model_section.config_overrides)

        log_config(log, "Model Config", config)

        # Load the model.
        if trainer_section.mixed_precision == "off":
            dtype = trainer_section.dtype
        else:
            dtype = torch.float32

        if self._has_checkpoint:
            log.info("Initializing the model.")
        else:
            log.info("Loading '{}' model on data parallel rank 0.", name)

        try:
            if gangs.dp.rank == 0 and not self._has_checkpoint:
                module = handler.load(
                    resolver, card, gangs, dtype, config, mmap=model_section.mmap
                )
            else:
                module = handler.create(
                    resolver, config, gangs, dtype, meta=handler.supports_meta
                )
        except NotSupportedError as ex:
            raise ModelLoadError(
                name, f"The '{name}' model cannot be constructed due to an unsupported operation. See the nested exception for details."  # fmt: skip
            ) from ex

        try:
            gangs.root.barrier()
        except GangError as ex:
            raise ModelLoadError(
                name, f"The collective barrier after the '{name}' model load operation has failed. See the nested exception for details."  # fmt: skip
            ) from ex

        module.train()

        if self._has_checkpoint:
            log.info("Model initialized.")
        else:
            log.info("Model loaded on data parallel rank 0.")

        return _BaseModel(name, module, config, handler)


@final
class _PathBasedModelLoader(_ModelLoader):
    _handlers: Provider[ModelHandler]
    _has_checkpoint: bool

    def __init__(self, handlers: Provider[ModelHandler], has_checkpoint: bool) -> None:
        self._handlers = handlers
        self._has_checkpoint = has_checkpoint

    @override
    def load(
        self,
        resolver: DependencyResolver,
        model_section: ModelSection,
        trainer_section: TrainerSection,
        gangs: Gangs,
    ) -> Model:
        family = model_section.family
        if family is None:
            raise ValueError("`model_section.family` must be specified.")

        path = model_section.path
        if path is None:
            raise ValueError("`model_section.path` must be specified.")

        name = path.name

        try:
            handler = self._handlers.resolve(family)
        except LookupError:
            raise UnknownModelFamilyError(family, name) from None

        if gangs.root.size != gangs.dp.size:
            if not handler.supports_model_parallelism:
                raise ModelParallelismNotSupportedError(name)

        arch = model_section.arch

        try:
            config = handler.get_arch_config(arch)
        except LookupError:
            if arch is not None:
                raise UnknownModelArchitectureError(arch, family, name) from None

            raise

        config = _apply_config_overrides(config, model_section.config_overrides)

        log_config(log, "Model Config", config)

        # Load the model.
        if trainer_section.mixed_precision == "off":
            dtype = trainer_section.dtype
        else:
            dtype = torch.float32

        if self._has_checkpoint:
            log.info("Initializing the model.")
        else:
            log.info("Loading '{}' model on data parallel rank 0.", name)

        try:
            if gangs.dp.rank == 0 and not self._has_checkpoint:
                try:
                    module = handler.load_from_path(
                        resolver,
                        path,
                        name,
                        config,
                        gangs,
                        dtype,
                        mmap=model_section.mmap,
                    )
                except FileNotFoundError:
                    raise ModelPathNotFoundError(name, path) from None
            else:
                module = handler.create(
                    resolver, config, gangs, dtype, meta=handler.supports_meta
                )
        except NotSupportedError as ex:
            raise ModelLoadError(
                name, f"The '{name}' model cannot be constructed due to an unsupported operation. See the nested exception for details."  # fmt: skip
            ) from ex

        try:
            gangs.root.barrier()
        except GangError as ex:
            raise ModelLoadError(
                name, f"The collective barrier after the '{name}' model load operation has failed. See the nested exception for details."  # fmt: skip
            ) from ex

        module.train()

        if self._has_checkpoint:
            log.info("Model initialized.")
        else:
            log.info("Model loaded on data parallel rank 0.")

        return _BaseModel(name, module, config, handler)


@final
class _NewModelLoader(_ModelLoader):
    _handlers: Provider[ModelHandler]
    _has_checkpoint: bool

    def __init__(self, handlers: Provider[ModelHandler], has_checkpoint: bool) -> None:
        self._handlers = handlers
        self._has_checkpoint = has_checkpoint

    @override
    def load(
        self,
        resolver: DependencyResolver,
        model_section: ModelSection,
        trainer_section: TrainerSection,
        gangs: Gangs,
    ) -> Model:
        family = model_section.family
        if family is None:
            raise ValueError("`model_section.family` must be specified.")

        name = "custom"

        try:
            handler = self._handlers.resolve(family)
        except LookupError:
            raise UnknownModelFamilyError(family, name) from None

        if gangs.root.size != gangs.dp.size:
            if not handler.supports_model_parallelism:
                raise ModelParallelismNotSupportedError(name)

        arch = model_section.arch

        try:
            config = handler.get_arch_config(arch)
        except LookupError:
            if arch is not None:
                raise UnknownModelArchitectureError(arch, family, name) from None

            raise

        config = _apply_config_overrides(config, model_section.config_overrides)

        log_config(log, "Model Config", config)

        # Create the model.
        if trainer_section.mixed_precision == "off":
            dtype = trainer_section.dtype
        else:
            dtype = torch.float32

        if self._has_checkpoint:
            log.info("Initializing the model.")
        else:
            log.info("Initializing the model on data parallel rank 0.")

        try:
            if gangs.dp.rank == 0 and not self._has_checkpoint:
                meta = False
            else:
                meta = handler.supports_meta

            module = handler.create(resolver, config, gangs, dtype, meta=meta)
        except NotSupportedError as ex:
            raise ModelLoadError(
                name, f"The '{name}' model cannot be constructed due to an unsupported operation. See the nested exception for details."  # fmt: skip
            ) from ex

        try:
            gangs.root.barrier()
        except GangError as ex:
            raise ModelLoadError(
                name, f"The collective barrier after the '{name}' model load operation has failed. See the nested exception for details."  # fmt: skip
            ) from ex

        module.train()

        if self._has_checkpoint:
            log.info("Model initialized.")
        else:
            log.info("Model initialized on data parallel rank 0.")

        return _BaseModel(
            name, module, config, handler, newly_initialized=not self._has_checkpoint
        )


def _apply_config_overrides(config: object, config_overrides: object) -> object:
    if config_overrides is None:
        return config

    try:
        config_overrides = structure(config_overrides, type(config), set_empty=True)
    except StructureError as ex:
        raise StructureError(
            "`model.config_overrides` cannot be structured. See the nested exception for details."
        ) from ex

    if not is_dataclass_instance(config):
        return config_overrides

    config_overrides = cast(DataClass, config_overrides)

    try:
        return merge_dataclass(config, config_overrides)
    except MergeError as ex:
        raise ContractError(
            "`config_overrides` cannot be merged with `config`. See the nested exception for details."
        ) from ex


def _save_model_metadata(resolver: DependencyResolver, model: Model) -> None:
    output_dir = get_output_dir(resolver)

    gangs = resolver.resolve(Gangs)

    file_system = resolver.resolve(FileSystem)

    yaml_dumper = resolver.resolve(YamlDumper)

    checkpoint_metadata_saver = CheckpointAssetMetadataSaver(
        output_dir.joinpath("checkpoints"), gangs, file_system, yaml_dumper
    )

    checkpoint_metadata_saver.save(model.handler.family, model.config)


@final
class _BaseModel(Model):
    _name: str
    _module: Module
    _config: object
    _handler: ModelHandler
    _newly_initialized: bool

    def __init__(
        self,
        name: str,
        module: Module,
        config: object,
        handler: ModelHandler,
        newly_initialized: bool = False,
    ) -> None:
        self._name = name
        self._module = module
        self._config = config
        self._handler = handler
        self._newly_initialized = newly_initialized

    @override
    def state_dict(self) -> dict[str, object]:
        return self._module.state_dict()

    @override
    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        self._module.load_state_dict(state_dict)

    @override
    def no_sync(self) -> ContextManager:
        return nullcontext()

    @override
    def clip_grad_norm(self, max_norm: float | None) -> Tensor:
        return clip_grad_norm(self._module, max_norm)

    @override
    def summon_full_parameters(self) -> ContextManager:
        return nullcontext()

    @property
    @override
    def name(self) -> str:
        return self._name

    @property
    @override
    def module(self) -> Module:
        return self._module

    @property
    @override
    def base_module(self) -> Module:
        return self._module

    @property
    @override
    def config(self) -> object:
        return self._config

    @property
    @override
    def handler(self) -> ModelHandler:
        return self._handler

    @property
    @override
    def newly_initialized(self) -> bool:
        return self._newly_initialized


def prepare_base_model(resolver: DependencyResolver, model: Model) -> None:
    trainer_section = get_config_section(resolver, "trainer", TrainerSection)

    ac = trainer_section.activation_checkpointing

    # Apply AC before torch.compile() so that min-cut partitioner can see the AC
    # information and avoid recomputing twice.
    if ac.mode == "layerwise":
        if not model.handler.supports_activation_checkpointing:
            raise ActivationCheckpointingNotSupportedError(model.name)

        model.handler.apply_activation_checkpointing(
            model.module, every_nth_layer=ac.every_nth_layer
        )

    model_section = get_config_section(resolver, "model", ModelSection)

    if model_section.compile:
        compile_model(model, model_section.compile_options)
