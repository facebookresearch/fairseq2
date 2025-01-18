# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import cast, final

from typing_extensions import override

from fairseq2.assets import AssetCardNotFoundError, AssetStore
from fairseq2.error import ContractError
from fairseq2.generation import Seq2SeqGeneratorHandler, SequenceGeneratorHandler
from fairseq2.metrics.recorders import MetricRecorderHandler
from fairseq2.models import ModelHandler, get_model_family
from fairseq2.optim import OptimizerHandler
from fairseq2.optim.lr_scheduler import LRSchedulerHandler
from fairseq2.recipes.config._dataclasses import (
    LRSchedulerSection,
    MetricsSection,
    ModelSection,
    OptimizerSection,
    Seq2SeqGeneratorSection,
    SequenceGeneratorSection,
)
from fairseq2.registry import Provider
from fairseq2.typing import DataClass, is_dataclass_instance, safe_cast
from fairseq2.utils.config import ConfigSectionHandler
from fairseq2.utils.dataclass import merge_dataclass
from fairseq2.utils.structured import StructureError, structure


@final
class ModelSectionHandler(ConfigSectionHandler):
    _asset_store: AssetStore
    _model_handlers: Provider[ModelHandler]

    def __init__(
        self, asset_store: AssetStore, model_handlers: Provider[ModelHandler]
    ) -> None:
        self._asset_store = asset_store
        self._model_handlers = model_handlers

    @override
    def process(self, section: object) -> None:
        section = safe_cast("section", section, ModelSection)

        if section.name is not None:
            try:
                card = self._asset_store.retrieve_card(section.name)
            except AssetCardNotFoundError:
                return

            section.family = get_model_family(card)

            try:
                handler = self._model_handlers.get(section.family)
            except LookupError:
                return

            model_config = handler.load_config(card)
        elif section.family is not None:
            try:
                handler = self._model_handlers.get(section.family)
            except LookupError:
                return

            model_config = handler.get_config(section.arch)
        else:
            return

        if section.config is not None:
            try:
                model_config_overrides = structure(
                    section.config, type(model_config), set_empty=True
                )
            except StructureError as ex:
                raise StructureError(
                    "`config` cannot be structured. See the nested exception for details."
                ) from ex

            if is_dataclass_instance(model_config):
                try:
                    model_config = merge_dataclass(
                        model_config, cast(DataClass, model_config_overrides)
                    )
                except ValueError as ex:
                    raise ContractError(
                        "`section.config` cannot be merged with the model configuration. See the nested exception for details."
                    ) from ex
            else:
                model_config = model_config_overrides

        section.config = model_config


@final
class OptimizerSectionHandler(ConfigSectionHandler):
    _optimizer_handlers: Provider[OptimizerHandler]

    def __init__(self, optimizer_handlers: Provider[OptimizerHandler]) -> None:
        self._optimizer_handlers = optimizer_handlers

    @override
    def process(self, section: object) -> None:
        section = safe_cast("section", section, OptimizerSection)

        try:
            optimizer_handler = self._optimizer_handlers.get(section.name)
        except LookupError:
            return

        try:
            section.config = structure(section.config, optimizer_handler.config_kls)
        except StructureError as ex:
            raise StructureError(
                "`config` cannot be structured. See the nested exception for details."
            ) from ex


@final
class LRSchedulerSectionHandler(ConfigSectionHandler):
    _lr_scheduler_handlers: Provider[LRSchedulerHandler]

    def __init__(self, lr_scheduler_handlers: Provider[LRSchedulerHandler]) -> None:
        self._lr_scheduler_handlers = lr_scheduler_handlers

    @override
    def process(self, section: object) -> None:
        section = safe_cast("section", section, LRSchedulerSection)

        if section.name is None:
            section.config = None
        else:
            try:
                lr_scheduler_handler = self._lr_scheduler_handlers.get(section.name)
            except LookupError:
                return

            try:
                section.config = structure(
                    section.config, lr_scheduler_handler.config_kls
                )
            except StructureError as ex:
                raise StructureError(
                    "`config` cannot be structured. See the nested exception for details."
                ) from ex


@final
class SequenceGeneratorSectionHandler(ConfigSectionHandler):
    _generator_handlers: Provider[SequenceGeneratorHandler]

    def __init__(self, generator_handlers: Provider[SequenceGeneratorHandler]) -> None:
        self._generator_handlers = generator_handlers

    @override
    def process(self, section: object) -> None:
        section = safe_cast("section", section, SequenceGeneratorSection)

        try:
            generator_handler = self._generator_handlers.get(section.name)
        except LookupError:
            return

        try:
            section.config = structure(section.config, generator_handler.config_kls)
        except StructureError as ex:
            raise StructureError(
                "`config` cannot be structured. See the nested exception for details."
            ) from ex


@final
class Seq2SeqGeneratorSectionHandler(ConfigSectionHandler):
    _generator_handlers: Provider[Seq2SeqGeneratorHandler]

    def __init__(self, generator_handlers: Provider[Seq2SeqGeneratorHandler]) -> None:
        self._generator_handlers = generator_handlers

    @override
    def process(self, section: object) -> None:
        section = safe_cast("section", section, Seq2SeqGeneratorSection)

        try:
            generator_handler = self._generator_handlers.get(section.name)
        except LookupError:
            return

        try:
            section.config = structure(section.config, generator_handler.config_kls)
        except StructureError as ex:
            raise StructureError(
                "`config` cannot be structured. See the nested exception for details."
            ) from ex


@final
class MetricsSectionHandler(ConfigSectionHandler):
    _recorder_handlers: Provider[MetricRecorderHandler]

    def __init__(self, recorder_handlers: Provider[MetricRecorderHandler]) -> None:
        self._recorder_handlers = recorder_handlers

    @override
    def process(self, section: object) -> None:
        section = safe_cast("section", section, MetricsSection)

        recorders = {}

        for name, config in section.recorders.items():
            try:
                recorder_handler = self._recorder_handlers.get(name)
            except LookupError:
                continue

            try:
                recorders[name] = structure(config, recorder_handler.config_kls)
            except StructureError as ex:
                raise StructureError(
                    f"`recorders.{name}.config` cannot be structured. See the nested exception for details."
                ) from ex

        section.recorders = recorders
