# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import nullcontext
from pathlib import Path
from typing import Mapping, cast, final

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
from fairseq2.checkpoint import (
    CheckpointError,
    CheckpointManager,
    CheckpointMetadataSaver,
    FileCheckpointMetadataSaver,
)
from fairseq2.config_registry import ConfigNotFoundError
from fairseq2.context import RuntimeContext
from fairseq2.error import ContractError, NotSupportedError
from fairseq2.gang import GangError, Gangs
from fairseq2.logging import log
from fairseq2.models import (
    InvalidModelTypeError,
    ModelConfigLoadError,
    ModelHandler,
    ModelLoadError,
    ShardedModelLoadError,
    UnknownModelArchitectureError,
    UnknownModelError,
    UnknownModelFamilyError,
    model_asset_card_error,
)
from fairseq2.nn.utils.gradient import clip_gradient_norm
from fairseq2.recipes import Model, RecipeError
from fairseq2.recipes.config import ModelSection, TorchCompileSection, TrainerSection
from fairseq2.recipes.utils.log import log_config, log_model
from fairseq2.registry import Provider
from fairseq2.typing import ContextManager, DataClass, is_dataclass_instance
from fairseq2.utils.merge import MergeError, merge_dataclass
from fairseq2.utils.structured import StructureError, structure
from fairseq2.utils.yaml import RuamelYamlDumper

# isort: split

from fairseq2.recipes.common._distributed import setup_data_parallel_model
from fairseq2.recipes.common._error import (
    ActivationCheckpointingNotSupportedError,
    InvalidModelPathError,
    ModelCompilationNotSupportedError,
    ModelParallelismNotSupportedError,
    ModelPathNotFoundError,
)


def setup_model(
    kls: type[Module],
    context: RuntimeContext,
    model_section: ModelSection,
    trainer_section: TrainerSection,
    output_dir: Path,
    gangs: Gangs,
    checkpoint_manager: CheckpointManager,
    static_graph: bool = True,
) -> Model:
    model = load_base_model(
        kls,
        context,
        model_section,
        trainer_section,
        output_dir,
        gangs,
        checkpoint_manager,
    )

    model = prepare_model(context, trainer_section, model)

    model = setup_data_parallel_model(
        context, trainer_section, model, gangs, static_graph
    )

    log_model(log, model.module, gangs)

    return model


def load_base_model(
    kls: type[Module],
    context: RuntimeContext,
    model_section: ModelSection,
    trainer_section: TrainerSection,
    output_dir: Path,
    gangs: Gangs,
    checkpoint_manager: CheckpointManager,
) -> Model:
    asset_store = context.asset_store

    file_system = context.file_system

    yaml_dumper = RuamelYamlDumper(file_system)

    checkpoint_metadata_saver = FileCheckpointMetadataSaver(
        output_dir.joinpath("checkpoints"), gangs, file_system, yaml_dumper
    )

    model_handlers = context.get_registry(ModelHandler)

    model_loader: ModelLoader

    if model_section.path is not None:
        model_loader = PathBasedModelLoader(
            kls, model_handlers, checkpoint_metadata_saver, checkpoint_manager
        )
    elif model_section.name is not None:
        model_loader = CardBasedModelLoader(
            kls,
            asset_store,
            model_handlers,
            checkpoint_metadata_saver,
            checkpoint_manager,
        )
    elif model_section.family is not None:
        model_loader = EmptyModelLoader(
            kls, model_handlers, checkpoint_metadata_saver, checkpoint_manager
        )
    else:
        raise ValueError(
            "Either `recipe_config.model.name` or `recipe_config.model.family` must be specified."
        )

    try:
        return model_loader.load(model_section, trainer_section, gangs)
    except ShardedModelLoadError:
        raise
    except ModelLoadError as ex:
        raise RecipeError(
            f"The '{ex.model_name}' model cannot be loaded. See the nested exception for details."
        ) from ex
    except AssetMetadataSaveError as ex:
        raise RecipeError(
            "The model card cannot be saved to the checkpoint directory. See the nested exception for details."
        ) from ex


class ModelLoader(ABC):
    @abstractmethod
    def load(
        self, model_section: ModelSection, trainer_section: TrainerSection, gangs: Gangs
    ) -> Model: ...


@final
class CardBasedModelLoader(ModelLoader):
    _kls: type[Module]
    _asset_store: AssetStore
    _model_handlers: Provider[ModelHandler]
    _checkpoint_metadata_saver: CheckpointMetadataSaver
    _checkpoint_manager: CheckpointManager

    def __init__(
        self,
        kls: type[Module],
        asset_store: AssetStore,
        model_handlers: Provider[ModelHandler],
        checkpoint_metadata_saver: CheckpointMetadataSaver,
        checkpoint_manager: CheckpointManager,
    ) -> None:
        self._kls = kls
        self._asset_store = asset_store
        self._model_handlers = model_handlers
        self._checkpoint_metadata_saver = checkpoint_metadata_saver
        self._checkpoint_manager = checkpoint_manager

    @override
    def load(
        self, model_section: ModelSection, trainer_section: TrainerSection, gangs: Gangs
    ) -> Model:
        model_name = model_section.name
        if model_name is None:
            raise ValueError("`recipe_config.model.name` must be specified.")

        try:
            card = self._asset_store.retrieve_card(model_name)
        except AssetCardNotFoundError:
            raise UnknownModelError(model_name) from None
        except AssetCardError as ex:
            raise model_asset_card_error(model_name) from ex

        try:
            model_family = card.field("model_family").as_(str)
        except AssetCardFieldNotFoundError:
            raise UnknownModelError(model_name) from None
        except AssetCardError as ex:
            raise model_asset_card_error(model_name) from ex

        try:
            handler = self._model_handlers.get(model_family)
        except LookupError:
            raise UnknownModelFamilyError(model_family, model_name) from None

        if not issubclass(handler.kls, self._kls):
            raise InvalidModelTypeError(model_name, handler.kls, self._kls)

        if gangs.root.size != gangs.dp.size:
            if not handler.supports_model_parallelism:
                raise ModelParallelismNotSupportedError(model_name)

        try:
            model_config = handler.load_config(card)
        except ModelConfigLoadError as ex:
            raise ModelLoadError(
                model_name, f"The '{model_name}' model configuration cannot be loaded. See the nested exception for details."  # fmt: skip
            ) from ex

        model_config = apply_config_overrides(
            model_config, model_section.config_overrides
        )

        log_config(log, "Model Config", model_config)

        # Load the model.
        if trainer_section.mixed_precision == "off":
            dtype = trainer_section.dtype
        else:
            dtype = torch.float32

        try:
            step_nr = self._checkpoint_manager.maybe_get_last_step_number(
                exclude_model_only=True
            )
        except CheckpointError:
            raise ModelLoadError(
                model_name, "The last training checkpoint cannot be retrieved. See the nested exception for details."  # fmt: skip
            )

        if step_nr is not None:
            model_name = f"checkpoint_step_{step_nr}"

            log.info("Last checkpoint found at step {}. Loading '{}' model on data parallel rank 0.", step_nr, model_name)  # fmt: skip
        else:
            log.info("Loading '{}' model on data parallel rank 0.", model_name)

        try:
            if gangs.dp.rank == 0:
                if step_nr is not None:
                    try:
                        model_path = self._checkpoint_manager.get_model_path(step_nr)
                    except CheckpointError:
                        raise ModelLoadError(
                            model_name, f"The path of the '{model_name}' model cannot be retrieved. See the nested exception for details."  # fmt: skip
                        )

                    try:
                        module = handler.load_from_path(
                            model_path,
                            model_name,
                            model_config,
                            gangs,
                            dtype,
                            mmap=model_section.mmap,
                        )
                    except FileNotFoundError:
                        raise ModelLoadError(
                            model_name, f"The '{model_name}' model cannot be found at the '{model_path}' path."  # fmt: skip
                        ) from None
                else:
                    module = handler.load(
                        card, gangs, dtype, model_config, mmap=model_section.mmap
                    )
            else:
                module = handler.create(
                    model_config, gangs, dtype, meta=handler.supports_meta
                )
        except NotSupportedError as ex:
            raise ModelLoadError(
                model_name, f"The '{model_name}' model cannot be constructed due to an unsupported operation. See the nested exception for details."  # fmt: skip
            ) from ex

        try:
            gangs.root.barrier()
        except GangError as ex:
            raise ModelLoadError(
                model_name, f"The collective barrier after the '{model_name}' model load operation has failed. See the nested exception for details."  # fmt: skip
            ) from ex

        self._checkpoint_metadata_saver.save(model_family, model_config)

        log.info("Model loaded on data parallel rank 0.")

        return BasicModel(model_name, module, model_config, handler)


@final
class PathBasedModelLoader(ModelLoader):
    _kls: type[Module]
    _model_handlers: Provider[ModelHandler]
    _checkpoint_metadata_saver: CheckpointMetadataSaver
    _checkpoint_manager: CheckpointManager

    def __init__(
        self,
        kls: type[Module],
        model_handlers: Provider[ModelHandler],
        checkpoint_metadata_saver: CheckpointMetadataSaver,
        checkpoint_manager: CheckpointManager,
    ) -> None:
        self._kls = kls
        self._model_handlers = model_handlers
        self._checkpoint_metadata_saver = checkpoint_metadata_saver
        self._checkpoint_manager = checkpoint_manager

    @override
    def load(
        self, model_section: ModelSection, trainer_section: TrainerSection, gangs: Gangs
    ) -> Model:
        model_family = model_section.family
        if model_family is None:
            raise ValueError("`recipe_config.model.family` must be specified.")

        model_path = model_section.path
        if model_path is None:
            raise ValueError("`recipe_config.model.path` must be specified.")

        model_path = self._format_as_sharded_path(model_path, gangs)

        model_name = "recipe"

        try:
            handler = self._model_handlers.get(model_family)
        except LookupError:
            raise UnknownModelFamilyError(model_family, model_name) from None

        if not issubclass(handler.kls, self._kls):
            raise InvalidModelTypeError(model_name, handler.kls, self._kls)

        if gangs.root.size != gangs.dp.size:
            if not handler.supports_model_parallelism:
                raise ModelParallelismNotSupportedError(model_name)

        model_arch = model_section.arch

        try:
            model_config = handler.get_arch_config(model_arch)
        except ConfigNotFoundError:
            if model_arch is not None:
                raise UnknownModelArchitectureError(
                    model_arch, model_family, model_name
                ) from None

            raise

        model_config = apply_config_overrides(
            model_config, model_section.config_overrides
        )

        log_config(log, "Model Config", model_config)

        # Load the model.
        if trainer_section.mixed_precision == "off":
            dtype = trainer_section.dtype
        else:
            dtype = torch.float32

        try:
            step_nr = self._checkpoint_manager.maybe_get_last_step_number(
                exclude_model_only=True
            )
        except CheckpointError:
            raise ModelLoadError(
                model_name, "The last training checkpoint cannot be retrieved. See the nested exception for details."  # fmt: skip
            )

        if step_nr is not None:
            model_name = f"checkpoint_step_{step_nr}"

            log.info("Last checkpoint found at step {}. Loading '{}' model on data parallel rank 0.", step_nr, model_name)  # fmt: skip
        else:
            log.info("Loading '{}' model on data parallel rank 0.", model_name)

        try:
            if gangs.dp.rank == 0:
                if step_nr is not None:
                    try:
                        model_path = self._checkpoint_manager.get_model_path(step_nr)
                    except CheckpointError:
                        raise ModelLoadError(
                            model_name, f"The path of the '{model_name}' model cannot be retrieved. See the nested exception for details."  # fmt: skip
                        )

                    try:
                        module = handler.load_from_path(
                            model_path,
                            model_name,
                            model_config,
                            gangs,
                            dtype,
                            mmap=model_section.mmap,
                        )
                    except FileNotFoundError:
                        raise ModelLoadError(
                            model_name, f"The '{model_name}' model cannot be found at the '{model_path}' path."  # fmt: skip
                        ) from None
                else:
                    try:
                        module = handler.load_from_path(
                            model_path,
                            model_name,
                            model_config,
                            gangs,
                            dtype,
                            mmap=model_section.mmap,
                        )
                    except FileNotFoundError:
                        raise ModelPathNotFoundError(model_name, model_path) from None
            else:
                module = handler.create(
                    model_config, gangs, dtype, meta=handler.supports_meta
                )
        except NotSupportedError as ex:
            raise ModelLoadError(
                model_name, f"The '{model_name}' model cannot be constructed due to an unsupported operation. See the nested exception for details."  # fmt: skip
            ) from ex

        try:
            gangs.root.barrier()
        except GangError as ex:
            raise ModelLoadError(
                model_name, f"The collective barrier after the '{model_name}' model load operation has failed. See the nested exception for details."  # fmt: skip
            ) from ex

        self._checkpoint_metadata_saver.save(model_family, model_config)

        log.info("Model loaded on data parallel rank 0.")

        return BasicModel(model_name, module, model_config, handler)

    @staticmethod
    def _format_as_sharded_path(model_path: Path, gangs: Gangs) -> Path:
        model_pathname = str(model_path)

        model_pathname = model_pathname.format_map({"shard_idx": gangs.tp.rank})

        try:
            return Path(model_pathname)
        except ValueError:
            raise InvalidModelPathError(model_pathname) from None


@final
class EmptyModelLoader(ModelLoader):
    _kls: type[Module]
    _model_handlers: Provider[ModelHandler]
    _checkpoint_metadata_saver: CheckpointMetadataSaver
    _checkpoint_manager: CheckpointManager

    def __init__(
        self,
        kls: type[Module],
        model_handlers: Provider[ModelHandler],
        checkpoint_metadata_saver: CheckpointMetadataSaver,
        checkpoint_manager: CheckpointManager,
    ) -> None:
        self._kls = kls
        self._model_handlers = model_handlers
        self._checkpoint_metadata_saver = checkpoint_metadata_saver
        self._checkpoint_manager = checkpoint_manager

    @override
    def load(
        self, model_section: ModelSection, trainer_section: TrainerSection, gangs: Gangs
    ) -> Model:
        model_family = model_section.family
        if model_family is None:
            raise ValueError("`recipe_config.model.family` must be specified.")

        model_name = "recipe"

        try:
            handler = self._model_handlers.get(model_family)
        except LookupError:
            raise UnknownModelFamilyError(model_family, model_name) from None

        if not issubclass(handler.kls, self._kls):
            raise InvalidModelTypeError(model_name, handler.kls, self._kls)

        if gangs.root.size != gangs.dp.size:
            if not handler.supports_model_parallelism:
                raise ModelParallelismNotSupportedError(model_name)

        model_arch = model_section.arch

        try:
            model_config = handler.get_arch_config(model_arch)
        except ConfigNotFoundError:
            if model_arch is not None:
                raise UnknownModelArchitectureError(
                    model_arch, model_family, model_name
                ) from None

            raise

        model_config = apply_config_overrides(
            model_config, model_section.config_overrides
        )

        log_config(log, "Model Config", model_config)

        # Create the model.
        if trainer_section.mixed_precision == "off":
            dtype = trainer_section.dtype
        else:
            dtype = torch.float32

        try:
            step_nr = self._checkpoint_manager.maybe_get_last_step_number(
                exclude_model_only=True
            )
        except CheckpointError:
            raise ModelLoadError(
                model_name, "The last training checkpoint cannot be retrieved. See the nested exception for details."  # fmt: skip
            )

        if step_nr is not None:
            model_name = f"checkpoint_step_{step_nr}"

            log.info("Last checkpoint found at step {}. Loading '{}' model on data parallel rank 0.", step_nr, model_name)  # fmt: skip

        try:
            if gangs.dp.rank == 0:
                if step_nr is not None:
                    try:
                        model_path = self._checkpoint_manager.get_model_path(step_nr)
                    except CheckpointError:
                        raise ModelLoadError(
                            model_name, f"The path of the '{model_name}' model cannot be retrieved. See the nested exception for details."  # fmt: skip
                        )

                    try:
                        module = handler.load_from_path(
                            model_path,
                            model_name,
                            model_config,
                            gangs,
                            dtype,
                            mmap=model_section.mmap,
                        )
                    except FileNotFoundError:
                        raise ModelLoadError(
                            model_name, f"The '{model_name}' model cannot be found at the '{model_path}' path."  # fmt: skip
                        ) from None
                else:
                    module = handler.create(model_config, gangs, dtype, meta=False)
            else:
                module = handler.create(
                    model_config, gangs, dtype, meta=handler.supports_meta
                )
        except NotSupportedError as ex:
            raise ModelLoadError(
                model_name, f"The '{model_name}' model cannot be constructed due to an unsupported operation. See the nested exception for details."  # fmt: skip
            ) from ex

        try:
            gangs.root.barrier()
        except GangError as ex:
            raise ModelLoadError(
                model_name, f"The collective barrier after the '{model_name}' model load operation has failed. See the nested exception for details."  # fmt: skip
            ) from ex

        self._checkpoint_metadata_saver.save(model_family, model_config)

        if step_nr is not None:
            log.info("Model loaded on data parallel rank 0.")

        return BasicModel(
            model_name,
            module,
            model_config,
            handler,
            empty_initialized=step_nr is None,
        )


def apply_config_overrides(config: object, config_overrides: object) -> object:
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


@final
class BasicModel(Model):
    _name: str
    _module: Module
    _config: object
    _handler: ModelHandler
    _empty_initialized: bool

    def __init__(
        self,
        name: str,
        module: Module,
        config: object,
        handler: ModelHandler,
        empty_initialized: bool = False,
    ) -> None:
        self._name = name
        self._module = module
        self._config = config
        self._handler = handler
        self._empty_initialized = empty_initialized

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
    def clip_gradient_norm(self, max_norm: float | None) -> Tensor:
        return clip_gradient_norm(self._module, max_norm)

    @override
    def summon_full_parameters(self) -> ContextManager:
        return nullcontext()

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
    def name(self) -> str:
        return self._name

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
    def empty_initialized(self) -> bool:
        return self._empty_initialized


def prepare_model(
    context: RuntimeContext, trainer_section: TrainerSection, model: Model
) -> Model:
    if trainer_section.activation_checkpointing:
        if not model.handler.supports_activation_checkpointing:
            raise ActivationCheckpointingNotSupportedError(model.name)

        model.handler.apply_activation_checkpointing(model.module)

    maybe_torch_compile_model(model, trainer_section.torch_compile)

    return model


def maybe_torch_compile_model(
    model: Model, torch_compile_section: TorchCompileSection
) -> None:
    if not torch_compile_section.enabled:
        return

    if not model.handler.supports_compilation:
        raise ModelCompilationNotSupportedError(model.name)

    log.info("Compiling '{}' model.", model.name)

    try:
        model.handler.compile(
            model.module,
            fullgraph=torch_compile_section.fullgraph,
            dynamic=torch_compile_section.dynamic,
            backend=torch_compile_section.backend,
            mode=torch_compile_section.mode,
            options=torch_compile_section.options,
        )
    except RuntimeError as ex:
        raise RecipeError(
            "torch.compile() has failed. See the nested exception for details."
        ) from ex
