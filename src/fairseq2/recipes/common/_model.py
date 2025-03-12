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
from fairseq2.data.text.tokenizers import (
    TextTokenizerLoadError,
    UnknownTextTokenizerError,
    resolve_text_tokenizer_reference,
    text_tokenizer_asset_card_error,
)
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
from fairseq2.nn.checkpointing import use_layerwise_activation_checkpointing
from fairseq2.nn.utils.gradient import clip_gradient_norm
from fairseq2.recipes import Model, RecipeError
from fairseq2.recipes.common._distributed import setup_data_parallel_model
from fairseq2.recipes.common._error import (
    InvalidCheckpointPathError,
    ModelCompilationNotSupportedError,
    ModelParallelismNotSupportedError,
    ModelPathNotFoundError,
)
from fairseq2.recipes.config import (
    ConfigSectionNotFoundError,
    ModelSection,
    TextTokenizerSection,
    TrainerSection,
    get_config_section,
)
from fairseq2.recipes.utils.log import log_config, log_model
from fairseq2.registry import Provider
from fairseq2.typing import ContextManager, DataClass, is_dataclass_instance
from fairseq2.utils.merge import MergeError, merge_dataclass
from fairseq2.utils.structured import StructureError, structure
from fairseq2.utils.yaml import StandardYamlDumper


def setup_model(
    kls: type[Module],
    context: RuntimeContext,
    recipe_config: object,
    output_dir: Path,
    gangs: Gangs,
    checkpoint_manager: CheckpointManager,
    static_graph: bool = True,
) -> Model:
    model = load_base_model(
        kls, context, recipe_config, output_dir, gangs, checkpoint_manager
    )

    model = prepare_model(context, recipe_config, model, gangs)

    model = setup_data_parallel_model(
        context, recipe_config, model, gangs, static_graph
    )

    log_model(log, model.module, gangs)

    return model


def load_base_model(
    kls: type[Module],
    context: RuntimeContext,
    recipe_config: object,
    output_dir: Path,
    gangs: Gangs,
    checkpoint_manager: CheckpointManager,
) -> Model:
    asset_store = context.asset_store

    file_system = context.file_system

    yaml_dumper = StandardYamlDumper(file_system)

    checkpoint_metadata_saver = FileCheckpointMetadataSaver(
        output_dir.joinpath("checkpoints"), gangs, file_system, yaml_dumper
    )

    model_card_saver = StandardModelCardSaver(asset_store, checkpoint_metadata_saver)

    model_handlers = context.get_registry(ModelHandler)

    model_loader: ModelLoader

    model_section = get_config_section(recipe_config, "model", ModelSection)
    if model_section.checkpoint is not None:
        model_loader = PathBasedModelLoader(
            kls, model_handlers, model_card_saver, checkpoint_manager
        )
    elif model_section.name is not None:
        model_loader = CardBasedModelLoader(
            kls, asset_store, model_handlers, model_card_saver, checkpoint_manager
        )
    elif model_section.family is not None:
        model_loader = ModelCreator(
            kls, model_handlers, model_card_saver, checkpoint_manager
        )
    else:
        raise ValueError(
            "Either `recipe_config.model.name` or `recipe_config.model.family` must be specified."
        )

    try:
        return model_loader.load(recipe_config, gangs)
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
    def load(self, recipe_config: object, gangs: Gangs) -> Model: ...


@final
class CardBasedModelLoader(ModelLoader):
    _kls: type[Module]
    _asset_store: AssetStore
    _model_handlers: Provider[ModelHandler]
    _card_saver: ModelCardSaver
    _checkpoint_manager: CheckpointManager

    def __init__(
        self,
        kls: type[Module],
        asset_store: AssetStore,
        model_handlers: Provider[ModelHandler],
        card_saver: ModelCardSaver,
        checkpoint_manager: CheckpointManager,
    ) -> None:
        self._kls = kls
        self._asset_store = asset_store
        self._model_handlers = model_handlers
        self._card_saver = card_saver
        self._checkpoint_manager = checkpoint_manager

    @override
    def load(self, recipe_config: object, gangs: Gangs) -> Model:
        model_section = get_config_section(recipe_config, "model", ModelSection)

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
            if not handler.supports_sharding:
                raise ModelParallelismNotSupportedError(model_name)

        try:
            model_config = handler.load_config(card)
        except ModelConfigLoadError as ex:
            raise ModelLoadError(
                model_name, f"The '{model_name}' model configuration cannot be loaded. See the nested exception for details."  # fmt: skip
            ) from ex

        model_config = apply_model_config_overrides(model_config, model_section.config)

        log_config(log, "Model Config", model_config)

        # Load the model.
        trainer_section = get_config_section(recipe_config, "trainer", TrainerSection)
        if trainer_section.mixed_precision is None:
            dtype = trainer_section.dtype
        else:
            dtype = torch.float32

        try:
            step_nr = self._checkpoint_manager.maybe_get_last_step_number()
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
                            model_path, model_name, model_config, gangs, dtype
                        )
                    except FileNotFoundError:
                        raise ModelLoadError(
                            model_name, f"The '{model_name}' model cannot be found at the '{model_path}' path."  # fmt: skip
                        ) from None
                else:
                    module = handler.load(card, gangs, dtype, model_config)
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

        log.info("Model loaded on data parallel rank 0.")

        model = LocalModel(model_name, module, model_config, handler)

        self._card_saver.save(recipe_config, model)

        return model


@final
class PathBasedModelLoader(ModelLoader):
    _kls: type[Module]
    _model_handlers: Provider[ModelHandler]
    _card_saver: ModelCardSaver
    _checkpoint_manager: CheckpointManager

    def __init__(
        self,
        kls: type[Module],
        model_handlers: Provider[ModelHandler],
        card_saver: ModelCardSaver,
        checkpoint_manager: CheckpointManager,
    ) -> None:
        self._kls = kls
        self._model_handlers = model_handlers
        self._card_saver = card_saver
        self._checkpoint_manager = checkpoint_manager

    @override
    def load(self, recipe_config: object, gangs: Gangs) -> Model:
        model_section = get_config_section(recipe_config, "model", ModelSection)

        model_family = model_section.family
        if model_family is None:
            raise ValueError("`recipe_config.model.family` must be specified.")

        model_path = model_section.checkpoint
        if model_path is None:
            raise ValueError("`recipe_config.model.checkpoint` must be specified.")

        model_path = self._format_as_sharded_path(model_path, gangs)

        model_name = "recipe"

        try:
            handler = self._model_handlers.get(model_family)
        except LookupError:
            raise UnknownModelFamilyError(model_family, model_name) from None

        if not issubclass(handler.kls, self._kls):
            raise InvalidModelTypeError(model_name, handler.kls, self._kls)

        if gangs.root.size != gangs.dp.size:
            if not handler.supports_sharding:
                raise ModelParallelismNotSupportedError(model_name)

        model_arch = model_section.arch

        try:
            model_config = handler.get_config(model_arch)
        except ConfigNotFoundError:
            if model_arch is not None:
                raise UnknownModelArchitectureError(
                    model_arch, model_family, model_name
                ) from None

            raise

        model_config = apply_model_config_overrides(model_config, model_section.config)

        log_config(log, "Model Config", model_config)

        # Load the model.
        trainer_section = get_config_section(recipe_config, "trainer", TrainerSection)
        if trainer_section.mixed_precision is None:
            dtype = trainer_section.dtype
        else:
            dtype = torch.float32

        try:
            step_nr = self._checkpoint_manager.maybe_get_last_step_number()
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
                            model_path, model_name, model_config, gangs, dtype
                        )
                    except FileNotFoundError:
                        raise ModelLoadError(
                            model_name, f"The '{model_name}' model cannot be found at the '{model_path}' path."  # fmt: skip
                        ) from None
                else:
                    try:
                        module = handler.load_from_path(
                            model_path, model_name, model_config, gangs, dtype
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

        log.info("Model loaded on data parallel rank 0.")

        model = LocalModel(model_name, module, model_config, handler)

        self._card_saver.save(recipe_config, model)

        return model

    @staticmethod
    def _format_as_sharded_path(model_path: Path, gangs: Gangs) -> Path:
        model_pathname = str(model_path)

        model_pathname = model_pathname.format_map({"shard_idx": gangs.tp.rank})

        try:
            return Path(model_pathname)
        except ValueError:
            raise InvalidCheckpointPathError(model_pathname) from None


@final
class ModelCreator(ModelLoader):
    _kls: type[Module]
    _model_handlers: Provider[ModelHandler]
    _card_saver: ModelCardSaver
    _checkpoint_manager: CheckpointManager

    def __init__(
        self,
        kls: type[Module],
        model_handlers: Provider[ModelHandler],
        card_saver: ModelCardSaver,
        checkpoint_manager: CheckpointManager,
    ) -> None:
        self._kls = kls
        self._model_handlers = model_handlers
        self._card_saver = card_saver
        self._checkpoint_manager = checkpoint_manager

    @override
    def load(self, recipe_config: object, gangs: Gangs) -> Model:
        model_section = get_config_section(recipe_config, "model", ModelSection)

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
            if not handler.supports_sharding:
                raise ModelParallelismNotSupportedError(model_name)

        model_arch = model_section.arch

        try:
            model_config = handler.get_config(model_arch)
        except ConfigNotFoundError:
            if model_arch is not None:
                raise UnknownModelArchitectureError(
                    model_arch, model_family, model_name
                ) from None

            raise

        model_config = apply_model_config_overrides(model_config, model_section.config)

        log_config(log, "Model Config", model_config)

        # Create the model.
        trainer_section = get_config_section(recipe_config, "trainer", TrainerSection)
        if trainer_section.mixed_precision is None:
            dtype = trainer_section.dtype
        else:
            dtype = torch.float32

        try:
            step_nr = self._checkpoint_manager.maybe_get_last_step_number()
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
                            model_path, model_name, model_config, gangs, dtype
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

        if step_nr is not None:
            log.info("Model loaded on data parallel rank 0.")

        model = LocalModel(
            model_name,
            module,
            model_config,
            handler,
            is_empty_initialized=step_nr is None,
        )

        self._card_saver.save(recipe_config, model)

        return model


def apply_model_config_overrides(model_config: object, overrides: object) -> object:
    if overrides is None:
        return model_config

    try:
        overrides = structure(overrides, type(model_config), set_empty=True)
    except StructureError as ex:
        raise StructureError(
            "`model.config` cannot be structured. See the nested exception for details."
        ) from ex

    if not is_dataclass_instance(model_config):
        return overrides

    overrides = cast(DataClass, overrides)

    try:
        return merge_dataclass(model_config, overrides)
    except MergeError as ex:
        raise ContractError(
            "`overrides` cannot be merged with `config`. See the nested exception for details."
        ) from ex


@final
class LocalModel(Model):
    _name: str
    _module: Module
    _config: object
    _handler: ModelHandler
    _is_empty_initialized: bool

    def __init__(
        self,
        name: str,
        module: Module,
        config: object,
        handler: ModelHandler,
        is_empty_initialized: bool = False,
    ) -> None:
        self._name = name
        self._module = module
        self._config = config
        self._handler = handler
        self._is_empty_initialized = is_empty_initialized

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
    def is_empty_initialized(self) -> bool:
        return self._is_empty_initialized


class ModelCardSaver(ABC):
    @abstractmethod
    def save(self, recipe_config: object, mode: Model) -> None: ...


@final
class StandardModelCardSaver(ModelCardSaver):
    _asset_store: AssetStore
    _checkpoint_metadata_saver: CheckpointMetadataSaver

    def __init__(
        self,
        asset_store: AssetStore,
        checkpoint_metadata_saver: CheckpointMetadataSaver,
    ) -> None:
        self._asset_store = asset_store
        self._checkpoint_metadata_saver = checkpoint_metadata_saver

    @override
    def save(self, recipe_config: object, model: Model) -> None:
        try:
            tokenizer_name = self._get_text_tokenizer_name(recipe_config)
        except TextTokenizerLoadError as ex:
            raise AssetMetadataSaveError(
                "The asset card of the model text tokenizer cannot be loaded. See the nested exception for details."
            ) from ex

        self._checkpoint_metadata_saver.save(
            model.handler.family, model.config, tokenizer_name
        )

    def _get_text_tokenizer_name(self, recipe_config: object) -> str | None:
        try:
            tokenizer_section = get_config_section(
                recipe_config, "text_tokenizer", TextTokenizerSection
            )
        except ConfigSectionNotFoundError:
            tokenizer_section = None

        if tokenizer_section is None:
            model_section = get_config_section(recipe_config, "model", ModelSection)

            tokenizer_name = model_section.name
            if tokenizer_name is None:
                return None
        else:
            tokenizer_name = tokenizer_section.name

        try:
            card = self._asset_store.retrieve_card(tokenizer_name)
        except AssetCardNotFoundError:
            if tokenizer_section is not None:
                raise UnknownTextTokenizerError(tokenizer_name) from None

            return None
        except AssetCardError as ex:
            raise text_tokenizer_asset_card_error(tokenizer_name) from ex

        try:
            card = resolve_text_tokenizer_reference(self._asset_store, card)
        except AssetCardError as ex:
            raise text_tokenizer_asset_card_error(tokenizer_name) from ex

        return card.name


def prepare_model(
    context: RuntimeContext, recipe_config: object, model: Model, gangs: Gangs
) -> Model:
    trainer_section = get_config_section(recipe_config, "trainer", TrainerSection)

    if trainer_section.activation_checkpointing:
        use_layerwise_activation_checkpointing(model.module)

    if trainer_section.torch_compile:
        if not model.handler.supports_compilation:
            raise ModelCompilationNotSupportedError(model.name)

        log.info("Compiling '{}' model.", model.name)

        model.handler.compile(model.module, model.config)

        log.info("Model compiled.")

    return model
