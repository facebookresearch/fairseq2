# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Callable
from typing import Sequence, final

import torch
from torch.optim import Optimizer

from fairseq2.checkpoint import NOOP_HG_EXPORTER, HuggingFaceExporter
from fairseq2.datasets import DataReader
from fairseq2.early_stopper import NOOP_EARLY_STOPPER
from fairseq2.evaluator import EvalUnit
from fairseq2.gang import Gangs
from fairseq2.logging import log
from fairseq2.metrics.recorders import (
    NOOP_METRIC_DESCRIPTOR,
    MetricDescriptor,
    MetricDescriptorRegistry,
)
from fairseq2.models.hg import HuggingFaceConverter
from fairseq2.optim.fp16_loss_scaler import (
    NOOP_FP16_LOSS_SCALER,
    Float16LossScaler,
    StandardFloat16LossScaler,
    supports_manual_grad_scaling,
)
from fairseq2.recipe.config import (
    CommonSection,
    ModelSection,
    RegimeSection,
    TrainerSection,
)
from fairseq2.recipe.error import (
    HuggingFaceNotSupportedError,
    ManualGradScalingNotSupportedError,
    MetricNotKnownError,
)
from fairseq2.recipe.internal.model import _ModelHolder
from fairseq2.runtime.lookup import Lookup
from fairseq2.trainer import BatchT, Trainer, TrainUnit
from fairseq2.utils.gc import (
    NOOP_GARBAGE_COLLECTOR,
    GarbageCollector,
    StandardGarbageCollector,
)
from fairseq2.validator import NOOP_VALIDATOR, StandardValidator, Validator


@final
class _TrainerFactory:
    def __init__(
        self,
        section: TrainerSection,
        regime_section: RegimeSection,
        common_section: CommonSection,
        gangs: Gangs,
        base_factory: Callable[..., Trainer],
    ) -> None:
        self._section = section
        self._regime_section = regime_section
        self._common_section = common_section
        self._gangs = gangs
        self._base_factory = base_factory

    def create(
        self,
        unit: TrainUnit[BatchT],
        data_reader: DataReader[BatchT],
        validator: Validator,
    ) -> Trainer:
        section = self._section

        regime_section = self._regime_section

        # TODO: Fix once we support static mixed precision on single device.
        mp_config = section.mixed_precision

        if mp_config.mode == "static":
            amp = self._gangs.root.size == 1 or section.data_parallelism != "fsdp"
            if amp:
                log.warning("As of this release, only FSDP supports static mixed precision training. Falling back to AMP/autocast.")  # fmt: skip
        else:
            amp = mp_config.mode == "auto"

        if self._gangs.root.device.type == "cpu":
            log.warning("Based on the environment setup, training will be run on CPU. If this is not intentional, check your job configuration (e.g. pass `--gpus-per-node` on Slurm).")  # fmt: skip

        seed = self._common_section.seed + 3

        return self._base_factory(
            unit=unit,
            data_reader=data_reader,
            amp=amp,
            amp_dtype=mp_config.dtype,
            no_sync_grad_accumulation=section.grad_accumulation.no_sync,
            max_grad_norm=section.max_grad_norm,
            grad_check=section.grad_check,
            anomaly_detection=section.anomaly_detection,
            seed=seed,
            max_num_steps=regime_section.num_steps,
            max_num_data_epochs=regime_section.num_data_epochs,
            validator=validator,
            validate_at_start=regime_section.validate_at_start,
            validate_after_n_steps=regime_section.validate_after_n_steps,
            validate_every_n_steps=regime_section.validate_every_n_steps,
            validate_after_n_data_epochs=regime_section.validate_after_n_data_epochs,
            validate_every_n_data_epochs=regime_section.validate_every_n_data_epochs,
            early_stopper=NOOP_EARLY_STOPPER,
            checkpoint_after_n_steps=regime_section.checkpoint_after_n_steps,
            checkpoint_every_n_steps=regime_section.checkpoint_every_n_steps,
            checkpoint_after_n_data_epochs=regime_section.checkpoint_after_n_data_epochs,
            checkpoint_every_n_data_epochs=regime_section.checkpoint_every_n_data_epochs,
            save_model_only=regime_section.save_model_only,
            keep_last_n_checkpoints=regime_section.keep_last_n_checkpoints,
            keep_best_n_checkpoints=regime_section.keep_best_n_checkpoints,
            keep_checkpoint_every_n_steps=regime_section.keep_checkpoint_every_n_steps,
            publish_metrics_after_n_steps=regime_section.publish_metrics_after_n_steps,
            publish_metrics_every_n_steps=regime_section.publish_metrics_every_n_steps,
            publish_metrics_after_n_data_epochs=regime_section.publish_metrics_after_n_data_epochs,
            publish_metrics_every_n_data_epochs=regime_section.publish_metrics_every_n_data_epochs,
        )


@final
class _ValidatorFactory:
    def __init__(
        self,
        section: TrainerSection,
        common_section: CommonSection,
        gangs: Gangs,
        standard_factory: Callable[..., StandardValidator],
    ) -> None:
        self._section = section
        self._common_section = common_section
        self._gangs = gangs
        self._standard_factory = standard_factory

    def create(
        self,
        valid_units: Sequence[EvalUnit[BatchT]] | None = None,
        valid_data_readers: Sequence[DataReader[BatchT]] | None = None,
    ) -> Validator:
        seed = self._common_section.seed + 3

        if valid_units:
            if valid_data_readers is None:
                raise ValueError(
                    "Both `valid_units` and `valid_data_readers` must be specified."
                )

            section = self._section

            # TODO: Fix once we support static mixed precision on single device.
            mp_config = section.mixed_precision

            if mp_config.mode == "static":
                amp = self._gangs.root.size == 1 or section.data_parallelism != "fsdp"
            else:
                amp = mp_config.mode == "auto"

            return self._standard_factory(
                units=valid_units,
                data_readers=valid_data_readers,
                amp=amp,
                amp_dtype=mp_config.dtype,
                seed=seed,
            )

        return NOOP_VALIDATOR


@final
class _Float16LossScalerFactory:
    def __init__(
        self,
        section: TrainerSection,
        model_section: ModelSection,
        optimizer: Optimizer,
        gangs: Gangs,
    ) -> None:
        self._section = section
        self._model_section = model_section
        self._optimizer = optimizer
        self._gangs = gangs

    def create(self) -> Float16LossScaler:
        mp_config = self._section.mixed_precision

        if mp_config.mode == "off":
            dtype = self._model_section.dtype
        else:
            dtype = mp_config.dtype

        if dtype != torch.float16:
            return NOOP_FP16_LOSS_SCALER

        gangs = self._gangs

        if gangs.sdp.size > 1:
            if not supports_manual_grad_scaling(self._optimizer):
                raise ManualGradScalingNotSupportedError()

        grad_accumulation = self._section.grad_accumulation.num_batches

        # Same formula as in fairseq.
        scale_window = max(int(2**14 / gangs.dp.size / grad_accumulation), 1)

        log.info("fp16 loss scale window set to {}.", scale_window)

        init_scale, min_scale = self._section.fp16_loss_scale

        return StandardFloat16LossScaler(
            gangs, init_scale=init_scale, scale_window=scale_window, min_scale=min_scale
        )


@final
class _HuggingFaceExporterFactory:
    def __init__(
        self,
        section: RegimeSection,
        model_holder: _ModelHolder,
        default_factory: Callable[[], HuggingFaceExporter],
        hg_converters: Lookup[HuggingFaceConverter],
    ) -> None:
        self._section = section
        self._model_holder = model_holder
        self._default_factory = default_factory
        self._hg_converter = hg_converters

    def create(self) -> HuggingFaceExporter:
        if not self._section.export_hugging_face:
            return NOOP_HG_EXPORTER

        hg_converter = self._hg_converter.maybe_get(self._model_holder.family.name)
        if hg_converter is None:
            raise HuggingFaceNotSupportedError()

        return self._default_factory()


@final
class _MaybeScoreMetricProvider:
    def __init__(
        self, section: RegimeSection, metric_descriptors: MetricDescriptorRegistry
    ) -> None:
        self._section = section
        self._metric_descriptors = metric_descriptors

    def maybe_get(self) -> MetricDescriptor:
        score_metric = self._section.score_metric
        if score_metric is None:
            return NOOP_METRIC_DESCRIPTOR

        descriptor = self._metric_descriptors.maybe_get(score_metric)
        if descriptor is None:
            raise MetricNotKnownError(score_metric)

        return descriptor


@final
class _GarbageCollectorFactory:
    def __init__(self, section: TrainerSection) -> None:
        self._section = section

    def create(self) -> GarbageCollector:
        if self._section.gc_every_n_steps is None:
            return NOOP_GARBAGE_COLLECTOR

        return StandardGarbageCollector(self._section.gc_every_n_steps)
