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
    CheckpointNotFoundError,
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
from fairseq2.error import ContractError, NotSupportedError, ProgramError
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
from fairseq2.models.compile import compile_model
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

    model = prepare_model(context, recipe_config, model, gangs)

    model = setup_data_parallel_model(
        context, recipe_config, model, gangs, static_graph
    )

    log_model(log, model, gangs)

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
        model = model_loader.load(recipe_config, gangs)
    except ShardedModelLoadError:
        raise
    except ModelLoadError as ex:
        raise ProgramError(
            f"The '{ex.model_name}' model cannot be loaded. See the nested exception for details."
        ) from ex
    except AssetMetadataSaveError as ex:
        raise ProgramError(
            "The model card cannot be saved to the checkpoint directory. See the nested exception for details."
        ) from ex

    return cast(ModelT, model)


class ModelLoader(ABC):
    @abstractmethod
    def load(self, recipe_config: object, gangs: Gangs) -> Module: ...


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
            raise InvalidModelTypeError(model_name, handler.kls, self._kls)

        if not handler.supports_sharding and gangs.root.size != gangs.dp.size:
            raise ModelParallelismNotSupportedError(model_family, model_name)

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
            step_nr, saved_model_path = self._checkpoint_manager.get_last_model_path()
        except CheckpointNotFoundError:
            step_nr, saved_model_path = 0, None
        except CheckpointError:
            raise ModelLoadError(
                model_name, "The path of the last checkpoint cannot be retrieved. See the nested exception for details."  # fmt: skip
            )

        if saved_model_path is not None:
            model_name = f"checkpoint_step_{step_nr}"

            log.info("Checkpoint found. Loading '{}' model on data parallel rank 0.", model_name)  # fmt: skip
        else:
            log.info("Loading '{}' model on data parallel rank 0.", model_name)

        try:
            if gangs.dp.rank == 0:
                if saved_model_path is not None:
                    try:
                        model = handler.load_from_path(
                            saved_model_path, model_name, model_config, gangs, dtype
                        )
                    except FileNotFoundError:
                        raise ModelLoadError(
                            model_name, f"The '{model_name}' model cannot be found at the '{saved_model_path}' path."  # fmt: skip
                        ) from None
                else:
                    model = handler.load(card, gangs, dtype, model_config)
            else:
                model = handler.create(
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
                model_name, f"The collective barrier after the load of the '{model_name}' model has failed. See the nested exception for details."  # fmt: skip
            ) from ex

        log.info("Model loaded on data parallel rank 0.")

        self._card_saver.save(recipe_config, model_family, model_config)

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
    def load(self, recipe_config: object, gangs: Gangs) -> Module:
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

        if not handler.supports_sharding and gangs.root.size != gangs.dp.size:
            raise ModelParallelismNotSupportedError(model_family, model_name)

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
            step_nr, saved_model_path = self._checkpoint_manager.get_last_model_path()
        except CheckpointNotFoundError:
            step_nr, saved_model_path = 0, None
        except CheckpointError:
            raise ModelLoadError(
                model_name, "The path of the last checkpoint cannot be retrieved. See the nested exception for details."  # fmt: skip
            )

        if saved_model_path is not None:
            model_name = f"checkpoint_step_{step_nr}"

            log.info("Checkpoint found. Loading '{}' model on data parallel rank 0.", model_name)  # fmt: skip
        else:
            log.info("Loading '{}' model on data parallel rank 0.", model_name)

        try:
            if gangs.dp.rank == 0:
                if saved_model_path is not None:
                    model_path = saved_model_path

                try:
                    model = handler.load_from_path(
                        model_path, model_name, model_config, gangs, dtype
                    )
                except FileNotFoundError:
                    if saved_model_path is None:
                        raise ModelNotFoundError(model_name, model_path) from None

                    raise ModelLoadError(
                        model_name, f"The '{model_name}' model cannot be found at the '{saved_model_path}' path."  # fmt: skip
                    ) from None
            else:
                model = handler.create(
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
                model_name, f"The collective barrier after the load of the '{model_name}' model has failed. See the nested exception for details."  # fmt: skip
            ) from ex

        log.info("Model loaded on data parallel rank 0.")

        self._card_saver.save(recipe_config, model_family, model_config)

        return model

    @staticmethod
    def _format_as_sharded_path(model_path: Path, gangs: Gangs) -> Path:
        model_pathname = str(model_path)

        model_pathname = model_pathname.format_map({"shard_idx": gangs.tp.rank})

        try:
            return Path(model_pathname)
        except ValueError:
            raise InvalidCheckpointPathError(model_pathname) from None


class InvalidCheckpointPathError(Exception):
    pathname: str

    def __init__(self, pathname: str) -> None:
        super().__init__(f"'{pathname}' does not represent a valid file system path.")

        self.pathname = pathname


class ModelNotFoundError(Exception):
    model_name: str
    path: Path

    def __init__(self, model_name: str, path: Path) -> None:
        super().__init__(
            f"The '{model_name}' model cannot be found at the '{path}' path."
        )

        self.model_name = model_name
        self.path = path


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
    def load(self, recipe_config: object, gangs: Gangs) -> Module:
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

        if not handler.supports_sharding and gangs.root.size != gangs.dp.size:
            raise ModelParallelismNotSupportedError(model_family, model_name)

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
            step_nr, saved_model_path = self._checkpoint_manager.get_last_model_path()
        except CheckpointNotFoundError:
            step_nr, saved_model_path = 0, None
        except CheckpointError:
            raise ModelLoadError(
                model_name, "The path of the last checkpoint cannot be retrieved. See the nested exception for details."  # fmt: skip
            )

        if saved_model_path is not None:
            model_name = f"checkpoint_step_{step_nr}"

            log.info("Checkpoint found. Loading '{}' model on data parallel rank 0.", model_name)  # fmt: skip

        try:
            if gangs.dp.rank == 0:
                if saved_model_path is not None:
                    try:
                        model = handler.load_from_path(
                            saved_model_path, model_name, model_config, gangs, dtype
                        )
                    except FileNotFoundError:
                        raise ModelLoadError(
                            model_name, f"The '{model_name}' model cannot be found at the '{saved_model_path}' path."  # fmt: skip
                        ) from None
                else:
                    model = handler.create(model_config, gangs, dtype, meta=False)
            else:
                model = handler.create(
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
                model_name, f"The collective barrier after the load of the '{model_name}' model has failed. See the nested exception for details."  # fmt: skip
            ) from ex

        if saved_model_path is not None:
            log.info("Model loaded on data parallel rank 0.")

        self._card_saver.save(recipe_config, model_family, model_config)

        return model


def apply_model_config_overrides(model_config: object, overrides: object) -> object:
    if overrides is None:
        return model_config

    try:
        structured_overrides = structure(overrides, type(model_config), set_empty=True)
    except StructureError as ex:
        raise StructureError(
            "`model.config` cannot be structured. See the nested exception for details."
        ) from ex

    if not is_dataclass_instance(model_config):
        return structured_overrides

    try:
        return merge_dataclass(model_config, cast(DataClass, structured_overrides))
    except MergeError as ex:
        raise ContractError(
            "`overrides` cannot be merged with `config`. See the nested exception for details."
        ) from ex


class ModelParallelismNotSupportedError(NotSupportedError):
    family: str
    model_name: str

    def __init__(self, family: str, model_name: str) -> None:
        super().__init__(
            f"The '{family}' family of the '{model_name}' model does not support non-data parallelism."
        )

        self.family = family
        self.model_name = model_name


class ModelCardSaver(ABC):
    @abstractmethod
    def save(
        self, recipe_config: object, model_family: str, model_config: object
    ) -> None: ...


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
    def save(
        self, recipe_config: object, model_family: str, model_config: object
    ) -> None:
        try:
            tokenizer_name = self._get_text_tokenizer_name(recipe_config)
        except TextTokenizerLoadError as ex:
            raise AssetMetadataSaveError(
                "The asset card of the model text tokenizer cannot be loaded. See the nested exception for details."
            ) from ex

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

    if trainer_section.torch_compile:
        log.info("Compiling model.")

        model = compile_model(model, gangs)

        log.info("Model compiled.")

    return model
