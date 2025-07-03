# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from pathlib import Path
from typing import final

from fairseq2.context import RuntimeContext
from fairseq2.gang import Gangs
from fairseq2.metrics.recorders import (
    CompositeMetricRecorder,
    MetricRecorder,
    MetricRecorderHandler,
    MetricRecordError,
    UnknownMetricRecorderError,
)
from fairseq2.recipes import RecipeError
from fairseq2.recipes.config import CommonSection
from fairseq2.registry import Provider
from fairseq2.utils.structured import StructureError


def create_metric_recorder(
    context: RuntimeContext,
    common_section: CommonSection,
    gangs: Gangs,
    output_dir: Path,
    hyper_params: object = None,
) -> MetricRecorder:
    recorder_handlers = context.get_registry(MetricRecorderHandler)

    creator = _MetricRecorderCreator(recorder_handlers)

    return creator.create(common_section, gangs, output_dir, hyper_params)


@final
class _MetricRecorderCreator:
    _metric_recorder_handlers: Provider[MetricRecorderHandler]

    def __init__(
        self, metric_recorder_handlers: Provider[MetricRecorderHandler]
    ) -> None:
        self._metric_recorder_handlers = metric_recorder_handlers

    def create(
        self,
        common_section: CommonSection,
        gangs: Gangs,
        output_dir: Path,
        hyper_params: object,
    ) -> MetricRecorder:
        recorders = []

        for recorder_name, recorder_config in common_section.metric_recorders.items():
            try:
                handler = self._metric_recorder_handlers.get(recorder_name)
            except LookupError:
                raise UnknownMetricRecorderError(recorder_name) from None

            if gangs.root.rank != 0:
                continue

            try:
                recorder = handler.create(output_dir, recorder_config, hyper_params)
            except StructureError as ex:
                raise StructureError(
                    f"`common.metric_recorders.{recorder_name}.config` cannot be structured. See the nested exception for details."
                ) from ex
            except MetricRecordError as ex:
                raise RecipeError(
                    f"The '{recorder_name}' metric recorder cannot be initialized. See the nested exception for details."
                ) from ex

            recorders.append(recorder)

        return CompositeMetricRecorder(recorders)
