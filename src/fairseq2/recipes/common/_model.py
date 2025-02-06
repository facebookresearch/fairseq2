# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TypeVar, cast, final

import torch
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
from fairseq2.context import RuntimeContext
from fairseq2.data.text.tokenizers import (
    TextTokenizerLoadError,
    UnknownTextTokenizerError,
    resolve_text_tokenizer_reference,
    text_tokenizer_asset_card_error,
)
from fairseq2.error import ContractError, NotSupportedError, ProgramError
from fairseq2.gang import GangError, Gangs
from fairseq2.logging import log
from fairseq2.models import (
    InvalidModelTypeError,
    ModelCheckpointNotFoundError,
    ModelConfigLoadError,
    ModelHandler,
    ModelLoadError,
    ModelParallelismNotSupportedError,
    ShardedModelLoadError,
    UnknownModelError,
    UnknownModelFamilyError,
    model_asset_card_error,
)
from fairseq2.nn.checkpointing import use_layerwise_activation_checkpointing
from fairseq2.recipes.common._distributed import setup_data_parallel_model
from fairseq2.recipes.config import (
    ConfigSectionNotFoundError,
    ModelSection,
    TextTokenizerSection,
    TrainerSection,
    get_config_section,
)
from fairseq2.recipes.utils.log import log_config, log_model
from fairseq2.registry import Provider
from fairseq2.typing import DataClass, is_dataclass_instance
from fairseq2.utils.merge import MergeError, merge_dataclass
from fairseq2.utils.structured import StructureError, structure
from fairseq2.utils.yaml import StandardYamlDumper

ModelT = TypeVar("ModelT", bound=Module)


def setup_model(
    kls: type[Module],
    context: RuntimeContext,
    recipe_config: object,
    output_dir: Path,
    gangs: Gangs,
    checkpoint_manager: CheckpointManager,
    static_graph: bool = True,
) -> Module:
    model = load_base_model(
        kls, context, recipe_config, output_dir, gangs, checkpoint_manager
    )

    log_model(log, model, gangs)
    model = setup_data_parallel_model(
        context, recipe_config, model, gangs, checkpoint_manager, static_graph
    )

    model = prepare_model(context, recipe_config, model, gangs)

    return model


def load_base_model(
    kls: type[ModelT],
    context: RuntimeContext,
    recipe_config: object,
    output_dir: Path,
    gangs: Gangs,
    checkpoint_manager: CheckpointManager,
) -> ModelT:
    asset_store = context.asset_store

    file_system = context.file_system

    yaml_dumper = StandardYamlDumper(file_system)

    checkpoint_metadata_saver = FileCheckpointMetadataSaver(
        output_dir.joinpath("checkpoints"), gangs, file_system, yaml_dumper
    )

    card_saver = FileModelCardSaver(asset_store, checkpoint_metadata_saver)

    model_handlers = context.get_registry(ModelHandler)

    model_section = get_config_section(recipe_config, "model", ModelSection)
    if model_section.checkpoint is not None:
        path_loader = PathBasedModelLoader(
            kls, model_handlers, checkpoint_manager, card_saver
        )

        model_name = "recipe"

        try:
            model = path_loader.load(recipe_config, model_name, gangs)
        except ModelParallelismNotSupportedError:
            raise
        except NotSupportedError as ex:
            raise ProgramError(
                f"The '{model_name}' model cannot be constructed due to an unsupported operation. See the nested exception for details."
            ) from ex
        except ModelCheckpointNotFoundError:
            raise
        except (ModelLoadError, TextTokenizerLoadError) as ex:
            raise ProgramError(
                f"The '{model_name}' model cannot be loaded. See the nested exception for details."
            ) from ex
        except AssetMetadataSaveError as ex:
            raise ProgramError(
                f"The '{model_name}' model card cannot be saved. See the nested exception for details."
            ) from ex
    elif model_section.name is not None:
        card_loader = CardBasedModelLoader(
            kls, asset_store, model_handlers, checkpoint_manager, card_saver
        )

        try:
            model = card_loader.load(recipe_config, gangs)
        except ModelParallelismNotSupportedError:
            raise
        except NotSupportedError as ex:
            raise ProgramError(
                f"The '{model_section.name}' model cannot be constructed due to an unsupported operation. See the nested exception for details."
            ) from ex
        except ShardedModelLoadError:
            raise
        except (ModelLoadError, TextTokenizerLoadError) as ex:
            raise ProgramError(
                f"The '{model_section.name}' model cannot be loaded. See the nested exception for details."
            ) from ex
        except AssetMetadataSaveError as ex:
            raise ProgramError(
                f"The '{model_section.name}' model card cannot be saved. See the nested exception for details."
            ) from ex
    elif model_section.family is not None:
        creator = ModelCreator(kls, model_handlers, card_saver)

        try:
            model = creator.create(recipe_config, gangs)
        except ModelParallelismNotSupportedError:
            raise
        except NotSupportedError as ex:
            raise ProgramError(
                "The model cannot be constructed due to an unsupported operation. See the nested exception for details."
            ) from ex
        except AssetMetadataSaveError as ex:
            raise ProgramError(
                "The model card cannot be saved. See the nested exception for details."
            ) from ex
    else:
        raise ValueError(
            "Either `recipe_config.model.name` or `recipe_config.model.family` must be specified."
        )

    return cast(ModelT, model)


@final
class CardBasedModelLoader:
    _kls: type[Module]
    _asset_store: AssetStore
    _model_handlers: Provider[ModelHandler]
    _checkpoint_manager: CheckpointManager
    _card_saver: ModelCardSaver

    def __init__(
        self,
        kls: type[Module],
        asset_store: AssetStore,
        model_handlers: Provider[ModelHandler],
        checkpoint_manager: CheckpointManager,
        card_saver: ModelCardSaver,
    ) -> None:
        self._kls = kls
        self._asset_store = asset_store
        self._model_handlers = model_handlers
        self._checkpoint_manager = checkpoint_manager
        self._card_saver = card_saver

    def load(self, recipe_config: object, gangs: Gangs) -> Module:
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
            raise InvalidModelTypeError(handler.kls, self._kls, model_name)

        try:
            model_config = handler.load_config(card)
        except ModelConfigLoadError as ex:
            raise ModelLoadError(
                model_name, f"The '{model_name}' model configuration cannot be loaded. See the nested exception for details."  # fmt: skip
            ) from ex

        model_config = apply_config_overrides(model_config, model_section.config)

        log_config(log, "Model Config", model_config)

        # Load the model.
        trainer_section = get_config_section(recipe_config, "trainer", TrainerSection)

        if trainer_section.mixed_precision is None:
            dtype = trainer_section.dtype
        else:
            dtype = torch.float32

        def create_model(meta: bool) -> Module:
            return handler.create(model_config, gangs, dtype, meta)

        # Shortcut if there is a checkpoint. No need to load the model.
        try:
            has_checkpoint = self._checkpoint_manager.has_checkpoint()
        except CheckpointError as ex:
            raise ModelLoadError(
                model_name, "The checkpoint state of the trainer cannot be checked. See the nested exception for details."  # fmt: skip
            ) from ex

        if has_checkpoint:
            model = create_model(handler.supports_meta)

            try:
                gangs.root.barrier()
            except GangError as ex:
                raise ModelLoadError(
                    model_name, f"The collective barrier after the load of the '{model_name}' model has failed. See the nested exception for details."  # fmt: skip
                ) from ex

            return model

        log.info("Loading '{}' model on data parallel rank 0 (per shard).", model_name)

        if gangs.dp.rank == 0:
            model = handler.load(card, gangs, dtype, model_config)
        else:
            model = create_model(handler.supports_meta)

        try:
            gangs.root.barrier()
        except GangError as ex:
            raise ModelLoadError(
                model_name, f"The collective barrier after the load of the '{model_name}' model has failed. See the nested exception for details."  # fmt: skip
            ) from ex

        log.info("Model loaded on data parallel rank 0.")

        self._card_saver.save(recipe_config, model_family, model_config)

        return model


@final
class PathBasedModelLoader:
    _kls: type[Module]
    _model_handlers: Provider[ModelHandler]
    _checkpoint_manager: CheckpointManager
    _card_saver: ModelCardSaver

    def __init__(
        self,
        kls: type[Module],
        model_handlers: Provider[ModelHandler],
        checkpoint_manager: CheckpointManager,
        card_saver: ModelCardSaver,
    ) -> None:
        self._kls = kls
        self._model_handlers = model_handlers
        self._checkpoint_manager = checkpoint_manager
        self._card_saver = card_saver

    def load(self, recipe_config: object, model_name: str, gangs: Gangs) -> Module:
        model_section = get_config_section(recipe_config, "model", ModelSection)

        model_family = model_section.family
        if model_family is None:
            raise ValueError("`recipe_config.model.family` must be specified.")

        checkpoint_path = model_section.checkpoint
        if checkpoint_path is None:
            raise ValueError("`recipe_config.model.checkpoint` must be specified.")

        checkpoint_path = self._format_as_sharded_path(checkpoint_path, gangs)

        try:
            handler = self._model_handlers.get(model_family)
        except LookupError:
            raise UnknownModelFamilyError(model_family, model_name) from None

        if not issubclass(handler.kls, self._kls):
            raise InvalidModelTypeError(handler.kls, self._kls, model_name)

        model_config = handler.get_config(model_section.arch)

        model_config = apply_config_overrides(model_config, model_section.config)

        log_config(log, "Model Config", model_config)

        # Load the model.
        trainer_section = get_config_section(recipe_config, "trainer", TrainerSection)

        if trainer_section.mixed_precision is None:
            dtype = trainer_section.dtype
        else:
            dtype = torch.float32

        def create_model(meta: bool) -> Module:
            return handler.create(model_config, gangs, dtype, meta)

        # Shortcut if there is a checkpoint. No need to load the model.
        try:
            has_checkpoint = self._checkpoint_manager.has_checkpoint()
        except CheckpointError as ex:
            raise ModelLoadError(
                model_name, "The checkpoint state of the trainer cannot be checked. See the nested exception for details."  # fmt: skip
            ) from ex

        if has_checkpoint:
            model = create_model(handler.supports_meta)

            try:
                gangs.root.barrier()
            except GangError as ex:
                raise ModelLoadError(
                    model_name, f"The collective barrier after the load of the '{model_name}' model has failed. See the nested exception for details."  # fmt: skip
                ) from ex

            return model

        log.info("Loading '{}' model on data parallel rank 0 (per shard).", model_name)

        if gangs.dp.rank == 0:
            model = handler.load_from_path(
                checkpoint_path, model_name, model_config, gangs, dtype
            )
        else:
            model = create_model(handler.supports_meta)

        try:
            gangs.root.barrier()
        except GangError as ex:
            raise ModelLoadError(
                model_name, f"The collective barrier after the load of the '{model_name}' model has failed. See the nested exception for details."  # fmt: skip
            ) from ex

        log.info("Model loaded on data parallel rank 0.")

        self._card_saver.save(recipe_config, model_family, model_config)

        return model

    @staticmethod
    def _format_as_sharded_path(checkpoint_path: Path, gangs: Gangs) -> Path:
        checkpoint_pathname = str(checkpoint_path)

        checkpoint_pathname = checkpoint_pathname.format_map(
            {"shard_idx": gangs.tp.rank}
        )

        try:
            return Path(checkpoint_pathname)
        except ValueError:
            raise InvalidCheckpointPathError(checkpoint_pathname) from None


class InvalidCheckpointPathError(Exception):
    pathname: str

    def __init__(self, pathname: str) -> None:
        super().__init__(f"'{pathname}' does not represent a valid file system path.")

        self.pathname = pathname


@final
class ModelCreator:
    _kls: type[Module]
    _model_handlers: Provider[ModelHandler]
    _card_saver: ModelCardSaver

    def __init__(
        self,
        kls: type[Module],
        model_handlers: Provider[ModelHandler],
        card_saver: ModelCardSaver,
    ) -> None:
        self._kls = kls
        self._model_handlers = model_handlers
        self._card_saver = card_saver

    def create(self, recipe_config: object, gangs: Gangs) -> Module:
        model_section = get_config_section(recipe_config, "model", ModelSection)

        model_family = model_section.family
        if model_family is None:
            raise ValueError("`recipe_config.model.family` must be specified.")

        try:
            handler = self._model_handlers.get(model_family)
        except LookupError:
            raise UnknownModelFamilyError(
                f"'{model_family}' is not a known model family."
            ) from None

        if not issubclass(handler.kls, self._kls):
            raise InvalidModelTypeError(handler.kls, self._kls)

        model_config = handler.get_config(model_section.arch)

        model_config = apply_config_overrides(model_config, model_section.config)

        log_config(log, "Model Config", model_config)

        # Create the model.
        trainer_section = get_config_section(recipe_config, "trainer", TrainerSection)

        if trainer_section.mixed_precision is None:
            dtype = trainer_section.dtype
        else:
            dtype = torch.float32

        model = handler.create(model_config, gangs, dtype, handler.supports_meta)

        self._card_saver.save(recipe_config, model_family, model_config)

        return model


def apply_config_overrides(config: object, overrides: object) -> object:
    if overrides is None:
        return config

    try:
        structured_overrides = structure(overrides, type(config), set_empty=True)
    except StructureError as ex:
        raise StructureError(
            "`model.config` cannot be structured. See the nested exception for details."
        ) from ex

    if not is_dataclass_instance(config):
        return structured_overrides

    try:
        return merge_dataclass(config, cast(DataClass, structured_overrides))
    except MergeError as ex:
        raise ContractError(
            "`overrides` cannot be merged with `config`. See the nested exception for details."
        ) from ex


class ModelCardSaver(ABC):
    @abstractmethod
    def save(
        self, recipe_config: object, model_family: str, model_config: object
    ) -> None: ...


@final
class FileModelCardSaver(ModelCardSaver):
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
    def save(
        self, recipe_config: object, model_family: str, model_config: object
    ) -> None:
        tokenizer_name = self._get_text_tokenizer_name(recipe_config)

        self._checkpoint_metadata_saver.save(model_family, model_config, tokenizer_name)

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
    context: RuntimeContext, recipe_config: object, model: Module, gangs: Gangs
) -> Module:
    trainer_section = get_config_section(recipe_config, "trainer", TrainerSection)

    if trainer_section.activation_checkpointing:
        use_layerwise_activation_checkpointing(model)

    log_model(log, model, gangs)

    if trainer_section.torch_compile:
        model = compile_model(context, recipe_config, model)

    return model


def compile_model(
    context: RuntimeContext, recipe_config: object, model: Module
) -> Module:
    # TODO: implement!
    return model
