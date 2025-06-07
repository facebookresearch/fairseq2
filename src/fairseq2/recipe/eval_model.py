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
from fairseq2.data_type import DataType
from fairseq2.error import InternalError, NotSupportedError
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
from fairseq2.nn.utils.module import broadcast_module, remove_parametrizations
from fairseq2.recipe.compile import _compile_model
from fairseq2.recipe.config import (
    ConfigOverrider,
    EvaluatorSection,
    GeneratorSection,
    ReferenceModelSection,
    TrainerSection,
    get_config_section,
)
from fairseq2.recipe.error import (
    ModelNotFoundError,
    ModelParallelismNotSupportedError,
)
from fairseq2.recipe.utils.log import log_model
from fairseq2.runtime.dependency import DependencyResolver
from fairseq2.utils.structured import StructureError, ValueConverter
from fairseq2.utils.validation import ObjectValidator, ValidationError


def _load_eval_model(resolver: DependencyResolver) -> ModelContext:
    evaluator_section = get_config_section(resolver, "evaluator", EvaluatorSection)

    return load_eval_model(
        resolver, "model", evaluator_section.dtype, evaluator_section.amp
    )


def _load_generator_model(resolver: DependencyResolver) -> ModelContext:
    generator_section = get_config_section(resolver, "generator", GeneratorSection)

    return load_eval_model(
        resolver, "model", generator_section.dtype, generator_section.amp
    )


def load_reference_model(
    resolver: DependencyResolver, section_name: str
) -> ModelContext:
    trainer_section = get_config_section(resolver, "trainer", TrainerSection)

    mp = trainer_section.mixed_precision != "off"

    return load_eval_model(resolver, section_name, trainer_section.dtype, mp)


def load_eval_model(
    resolver: DependencyResolver, section_name: str, dtype: DataType, mp: bool
) -> ModelContext:
    section = get_config_section(resolver, section_name, ReferenceModelSection)

    if section.path is not None:
        return _load_eval_model_from_path(resolver, section_name, section, dtype, mp)

    if section.name is not None:
        return _load_eval_model_from_card(resolver, section_name, section, dtype, mp)

    raise InternalError("`section.name` or `section.path` are both `None`.")


def _load_eval_model_from_card(
    resolver: DependencyResolver,
    section_name: str,
    section: ReferenceModelSection,
    dtype: DataType,
    mp: bool,
) -> ModelContext:
    value_converter = resolver.resolve(ValueConverter)

    asset_store = resolver.resolve(AssetStore)

    handlers = resolver.get_provider(ModelFamilyHandler)

    gangs = resolver.resolve(Gangs)

    validator = resolver.resolve(ObjectValidator)

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
            f"`{section_name}.config_overrides` cannot be structured. See the nested exception for details."
        ) from ex

    try:
        validator.validate(config)
    except ValidationError as ex:
        raise ValidationError(
            ex.result, field=f"{section_name}.config_overrides"
        ) from None

    # Load the model.
    if mp:
        dtype = torch.float32

    log.info("Loading '{}' model on data parallel rank 0.", name)

    try:
        if gangs.dp.rank == 0:
            module = handler.load_model(
                resolver,
                card,
                gangs,
                dtype,
                config=config,
                mmap=section.mmap,
            )
        else:
            module = handler.create_new_model(
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

    module.eval()

    return StandardModelContext(name, module, config, handler)


def _load_eval_model_from_path(
    resolver: DependencyResolver,
    section_name: str,
    section: ReferenceModelSection,
    dtype: DataType,
    mp: bool,
) -> ModelContext:
    value_converter = resolver.resolve(ValueConverter)

    handlers = resolver.get_provider(ModelFamilyHandler)

    gangs = resolver.resolve(Gangs)

    validator = resolver.resolve(ObjectValidator)

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
            f"`{section_name}.config_overrides` cannot be structured. See the nested exception for details."
        ) from ex

    try:
        validator.validate(config)
    except ValidationError as ex:
        raise ValidationError(
            ex.result, field=f"{section_name}.config_overrides"
        ) from None

    # Load the model.
    if mp:
        dtype = torch.float32

    log.info("Loading '{}' model on data parallel rank 0.", name)

    try:
        if gangs.dp.rank == 0:
            module = handler.load_model_from_path(
                resolver,
                path,
                config,
                gangs,
                dtype,
                mmap=section.mmap,
            )
        else:
            module = handler.create_new_model(
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
    except FileNotFoundError as ex:
        raise ModelNotFoundError(path) from ex

    gangs.root.barrier()

    log.info("Model loaded on data parallel rank 0.")

    module.eval()

    return StandardModelContext(name, module, config, handler)


def _prepare_eval_model(
    resolver: DependencyResolver, model_context: ModelContext
) -> ModelContext:
    section = get_config_section(resolver, "model", ReferenceModelSection)

    gangs = resolver.resolve(Gangs)

    _broadcast_model(resolver, model_context)

    remove_parametrizations(model_context.model)

    if section.compile:
        _compile_model(model_context, section.compile_options)

    log_model(model_context.model, gangs)

    return model_context


def _broadcast_model(resolver: DependencyResolver, model_context: ModelContext) -> None:
    gangs = resolver.resolve(Gangs)

    if gangs.dp.size == 1:
        return

    log.info("Broadcasting '{}' model to all processes.", model_context.name)

    broadcast_module(model_context.model, gangs.dp)

    log.info("Model broadcasted.")
