# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch

from fairseq2.assets import (
    AssetCardFieldNotFoundError,
    AssetNotFoundError,
    AssetStore,
)
from fairseq2.checkpoint import CheckpointAssetMetadataSaver
from fairseq2.error import InternalError, NotSupportedError
from fairseq2.file_system import FileSystem
from fairseq2.gang import Gangs
from fairseq2.logging import log
from fairseq2.model.context import ModelContext, StandardModelContext
from fairseq2.models import (
    ModelConfigLoadError,
    ModelFamilyHandler,
    ModelLoadError,
    UnknownModelError,
    UnknownModelFamilyError,
)
from fairseq2.recipe.checkpoint import _check_has_checkpoint
from fairseq2.recipe.compile import _compile_model
from fairseq2.recipe.config import (
    ConfigOverrider,
    ModelSection,
    TrainerSection,
    get_config_section,
    get_output_dir,
)
from fairseq2.recipe.data_parallel_model import _create_data_parallel_model
from fairseq2.recipe.error import (
    ActivationCheckpointingNotSupportedError,
    ModelInitializationError,
    ModelNotFoundError,
    ModelParallelismNotSupportedError,
)
from fairseq2.recipe.utils.log import log_config, log_model
from fairseq2.runtime.dependency import DependencyResolver
from fairseq2.utils.structured import StructureError, ValueConverter
from fairseq2.utils.validation import ObjectValidator, ValidationError
from fairseq2.utils.yaml import YamlDumper


def _create_or_load_model(resolver: DependencyResolver) -> ModelContext:
    section = get_config_section(resolver, "model", ModelSection)

    if section.path is not None:
        model_context = _load_model_from_path(resolver)
    elif section.name is not None:
        model_context = _load_model_from_card(resolver)
    elif section.family is not None:
        model_context = _create_new_model(resolver)
    else:
        raise InternalError("`section.name` or `section.family` are both `None`.")

    _save_model_metadata(resolver, model_context)

    return model_context


def _load_model_from_card(resolver: DependencyResolver) -> ModelContext:
    section = get_config_section(resolver, "model", ModelSection)

    trainer_section = get_config_section(resolver, "trainer", TrainerSection)

    value_converter = resolver.resolve(ValueConverter)

    asset_store = resolver.resolve(AssetStore)

    handlers = resolver.get_provider(ModelFamilyHandler)

    gangs = resolver.resolve(Gangs)

    validator = resolver.resolve(ObjectValidator)

    has_checkpoint = _check_has_checkpoint(resolver)

    config_overrider = ConfigOverrider(value_converter)

    name = section.name
    if name is None:
        raise InternalError("`section.name` is `None`.")

    try:
        card = asset_store.retrieve_card(name)
    except AssetNotFoundError:
        raise UnknownModelError(name) from None

    try:
        family = card.field("model_family").as_(str)
    except AssetCardFieldNotFoundError:
        raise UnknownModelError(name) from None

    try:
        try:
            handler = handlers.get(family)
        except LookupError:
            raise UnknownModelFamilyError(family) from None
    except UnknownModelFamilyError as ex:
        raise ModelLoadError(
            name, f"The '{family}' family of the '{name}' model is not known."  # fmt: skip
        ) from ex

    if gangs.root.size != gangs.dp.size:
        if not handler.supports_model_parallelism:
            raise ModelParallelismNotSupportedError(name)

    try:
        config = handler.load_model_config(card)
    except ModelConfigLoadError as ex:
        raise ModelLoadError(
            name, f"The configuration of the '{name}' model cannot be loaded. See the nested exception for details."  # fmt: skip
        ) from ex

    try:
        config = config_overrider.apply_overrides(config, section.config_overrides)
    except StructureError as ex:
        raise StructureError(
            "`model.config_overrides` cannot be structured. See the nested exception for details."
        ) from ex

    try:
        validator.validate(config)
    except ValidationError as ex:
        raise ValidationError(ex.result, field="model.config_overrides") from None

    log_config("Model Config", config)

    # Load the model.
    if trainer_section.mixed_precision == "off":
        dtype = trainer_section.dtype
    else:
        dtype = torch.float32

    if has_checkpoint:
        if not handler.supports_meta:
            log.info("Initializing '{}' model.", name)

        try:
            model = handler.create_new_model(
                resolver, config, gangs, dtype, meta=handler.supports_meta
            )
        except ValueError as ex:
            raise ModelInitializationError(
                name, f"The '{name}' model does not have a valid configuration. See the nested exception for details."  # fmt: skip
            ) from ex
        except NotSupportedError as ex:
            raise ModelInitializationError(
                name, f"The '{name}' model cannot be constructed due to an unsupported operation. See the nested exception for details."  # fmt: skip
            ) from ex

        if not handler.supports_meta:
            log.info("Model initialized.")
    else:
        log.info("Loading '{}' model on data parallel rank 0.", name)

        try:
            if gangs.dp.rank == 0:
                model = handler.load_model(
                    resolver,
                    card,
                    gangs,
                    dtype,
                    config=config,
                    mmap=section.mmap,
                )
            else:
                model = handler.create_new_model(
                    resolver, config, gangs, dtype, meta=handler.supports_meta
                )
        except ValueError as ex:
            raise ModelLoadError(
                name, f"The '{name}' model does not have a valid configuration. See the nested exception for details."  # fmt: skip
            ) from ex
        except NotSupportedError as ex:
            raise ModelLoadError(
                name, f"The '{name}' model cannot be constructed due to an unsupported operation. See the nested exception for details."  # fmt: skip
            ) from ex

        gangs.root.barrier()

        log.info("Model loaded on data parallel rank 0.")

    model.train()

    return StandardModelContext(name, model, config, handler)


def _load_model_from_path(resolver: DependencyResolver) -> ModelContext:
    section = get_config_section(resolver, "model", ModelSection)

    trainer_section = get_config_section(resolver, "trainer", TrainerSection)

    value_converter = resolver.resolve(ValueConverter)

    handlers = resolver.get_provider(ModelFamilyHandler)

    gangs = resolver.resolve(Gangs)

    validator = resolver.resolve(ObjectValidator)

    has_checkpoint = _check_has_checkpoint(resolver)

    config_overrider = ConfigOverrider(value_converter)

    path = section.path
    if path is None:
        raise InternalError("`section.path` is `None`.")

    family = section.family
    if family is None:
        raise InternalError("`section.family` is `None`.")

    name = str(path)

    try:
        handler = handlers.get(family)
    except LookupError:
        raise UnknownModelFamilyError(family) from None

    if gangs.root.size != gangs.dp.size:
        if not handler.supports_model_parallelism:
            raise ModelParallelismNotSupportedError(name)

    arch = section.arch

    config = handler.get_arch_config(arch)

    try:
        config = config_overrider.apply_overrides(config, section.config_overrides)
    except StructureError as ex:
        raise StructureError(
            "`model.config_overrides` cannot be structured. See the nested exception for details."
        ) from ex

    try:
        validator.validate(config)
    except ValidationError as ex:
        raise ValidationError(ex.result, field="model.config_overrides") from None

    log_config("Model Config", config)

    # Load the model.
    if trainer_section.mixed_precision == "off":
        dtype = trainer_section.dtype
    else:
        dtype = torch.float32

    if has_checkpoint:
        if not handler.supports_meta:
            log.info("Initializing '{}' model.", name)

        try:
            model = handler.create_new_model(
                resolver, config, gangs, dtype, meta=handler.supports_meta
            )
        except ValueError as ex:
            raise ModelInitializationError(
                name, f"The '{name}' model does not have a valid configuration. See the nested exception for details."  # fmt: skip
            ) from ex
        except NotSupportedError as ex:
            raise ModelInitializationError(
                name, f"The '{name}' model cannot be constructed due to an unsupported operation. See the nested exception for details."  # fmt: skip
            ) from ex

        if not handler.supports_meta:
            log.info("Model initialized.")
    else:
        log.info("Loading '{}' model on data parallel rank 0.", name)

        try:
            if gangs.dp.rank == 0:
                model = handler.load_model_from_path(
                    resolver,
                    path,
                    config,
                    gangs,
                    dtype,
                    mmap=section.mmap,
                )
            else:
                model = handler.create_new_model(
                    resolver, config, gangs, dtype, meta=handler.supports_meta
                )
        except ValueError as ex:
            raise ModelLoadError(
                name, f"The '{name}' model does not have a valid configuration. See the nested exception for details."  # fmt: skip
            ) from ex
        except NotSupportedError as ex:
            raise ModelLoadError(
                name, f"The '{name}' model cannot be constructed due to an unsupported operation. See the nested exception for details."  # fmt: skip
            ) from ex
        except FileNotFoundError:
            raise ModelNotFoundError(path) from None

        gangs.root.barrier()

        log.info("Model loaded on data parallel rank 0.")

    model.train()

    return StandardModelContext(name, model, config, handler)


def _create_new_model(resolver: DependencyResolver) -> ModelContext:
    section = get_config_section(resolver, "model", ModelSection)

    trainer_section = get_config_section(resolver, "trainer", TrainerSection)

    value_converter = resolver.resolve(ValueConverter)

    handlers = resolver.get_provider(ModelFamilyHandler)

    gangs = resolver.resolve(Gangs)

    validator = resolver.resolve(ObjectValidator)

    has_checkpoint = _check_has_checkpoint(resolver)

    config_overrider = ConfigOverrider(value_converter)

    family = section.family
    if family is None:
        raise InternalError("`section.family` is `None`.")

    name = "train"

    try:
        handler = handlers.get(family)
    except LookupError:
        raise UnknownModelFamilyError(family) from None

    if gangs.root.size != gangs.dp.size:
        if not handler.supports_model_parallelism:
            raise ModelParallelismNotSupportedError(name)

    arch = section.arch

    config = handler.get_arch_config(arch)

    try:
        config = config_overrider.apply_overrides(config, section.config_overrides)
    except StructureError as ex:
        raise StructureError(
            "`model.config_overrides` cannot be structured. See the nested exception for details."
        ) from ex

    try:
        validator.validate(config)
    except ValidationError as ex:
        raise ValidationError(ex.result, field="model.config_overrides") from None

    log_config("Model Config", config)

    # Create the model.
    if trainer_section.mixed_precision == "off":
        dtype = trainer_section.dtype
    else:
        dtype = torch.float32

    if has_checkpoint:
        if not handler.supports_meta:
            log.info("Initializing '{}' model.", name)

        try:
            model = handler.create_new_model(
                resolver, config, gangs, dtype, meta=handler.supports_meta
            )
        except ValueError as ex:
            raise ModelInitializationError(
                name, f"The '{name}' model does not have a valid configuration. See the nested exception for details."  # fmt: skip
            ) from ex
        except NotSupportedError as ex:
            raise ModelInitializationError(
                name, f"The '{name}' model cannot be constructed due to an unsupported operation. See the nested exception for details."  # fmt: skip
            ) from ex

        if not handler.supports_meta:
            log.info("Model initialized.")
    else:
        if not handler.supports_meta:
            log.info("Initializing '{}' model.", name)
        else:
            log.info("Initializing '{}' model on data parallel rank 0.", name)

        try:
            if gangs.dp.rank == 0:
                meta = False
            else:
                meta = handler.supports_meta

            model = handler.create_new_model(resolver, config, gangs, dtype, meta=meta)
        except ValueError as ex:
            raise ModelLoadError(
                name, f"The '{name}' model does not have a valid configuration. See the nested exception for details."  # fmt: skip
            ) from ex
        except NotSupportedError as ex:
            raise ModelLoadError(
                name, f"The '{name}' model cannot be constructed due to an unsupported operation. See the nested exception for details."  # fmt: skip
            ) from ex

        gangs.root.barrier()

        if not handler.supports_meta:
            log.info("Model initialized.")
        else:
            log.info("Model initialized on data parallel rank 0.")

    model.train()

    return StandardModelContext(
        name, model, config, handler, newly_initialized=not has_checkpoint
    )


def _save_model_metadata(
    resolver: DependencyResolver, model_context: ModelContext
) -> None:
    gangs = resolver.resolve(Gangs)

    file_system = resolver.resolve(FileSystem)

    yaml_dumper = resolver.resolve(YamlDumper)

    value_converter = resolver.resolve(ValueConverter)

    output_dir = get_output_dir(resolver)

    checkpoint_metadata_saver = CheckpointAssetMetadataSaver(
        file_system, yaml_dumper, value_converter
    )

    checkpoint_dir = output_dir.joinpath("checkpoints")

    checkpoint_metadata_saver.save(
        checkpoint_dir, gangs, model_context.handler.family, model_context.config
    )


def _prepare_model(
    resolver: DependencyResolver, model_context: ModelContext
) -> ModelContext:
    section = get_config_section(resolver, "model", ModelSection)

    trainer_section = get_config_section(resolver, "trainer", TrainerSection)

    gangs = resolver.resolve(Gangs)

    ac = trainer_section.activation_checkpointing

    # Apply AC before torch.compile() so that min-cut partitioner can see the AC
    # information and avoid recomputing twice.
    if ac.mode == "layerwise":
        if not model_context.handler.supports_activation_checkpointing:
            raise ActivationCheckpointingNotSupportedError(model_context.name)

        model_context.handler.apply_activation_checkpointing(
            model_context.model, every_nth_layer=ac.every_nth_layer
        )

    if section.compile:
        _compile_model(model_context, section.compile_options)

    model_context = _create_data_parallel_model(resolver, model_context)

    log_model(model_context.model, gangs)

    return model_context
