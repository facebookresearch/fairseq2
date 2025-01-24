# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.context import RuntimeContext
from fairseq2.generation import (
    BeamSearchAlgorithmHandler,
    BeamSearchAlgorithmSection,
    BeamSearchAlgorithmSectionHandler,
    SamplerHandler,
    SamplerSection,
    SamplerSectionHandler,
    Seq2SeqGeneratorHandler,
    SequenceGeneratorHandler,
)
from fairseq2.metrics.recorders import MetricRecorderHandler
from fairseq2.models import ModelHandler
from fairseq2.optim import OptimizerHandler
from fairseq2.optim.lr_scheduler import LRSchedulerHandler
from fairseq2.recipes.config import (
    LRSchedulerSection,
    LRSchedulerSectionHandler,
    MetricsSection,
    MetricsSectionHandler,
    ModelSection,
    ModelSectionHandler,
    OptimizerSection,
    OptimizerSectionHandler,
    Seq2SeqGeneratorSection,
    Seq2SeqGeneratorSectionHandler,
    SequenceGeneratorSection,
    SequenceGeneratorSectionHandler,
)
from fairseq2.recipes.lm import (
    POCriterionSection,
    POCriterionSectionHandler,
    POFinetuneUnitHandler,
)
from fairseq2.utils.config import ConfigSectionHandler


def _register_config_sections(context: RuntimeContext) -> None:
    registry = context.get_registry(ConfigSectionHandler)

    handler: ConfigSectionHandler

    # Beam Search Algorithm
    bs_algorithm_handlers = context.get_registry(BeamSearchAlgorithmHandler)

    handler = BeamSearchAlgorithmSectionHandler(bs_algorithm_handlers)

    registry.register(BeamSearchAlgorithmSection, handler)

    # Learning Rate Scheduler
    lr_scheduler_handlers = context.get_registry(LRSchedulerHandler)

    handler = LRSchedulerSectionHandler(lr_scheduler_handlers)

    registry.register(LRSchedulerSection, handler)

    # Metrics
    recorder_handlers = context.get_registry(MetricRecorderHandler)

    handler = MetricsSectionHandler(recorder_handlers)

    registry.register(MetricsSection, handler)

    # Model
    model_handlers = context.get_registry(ModelHandler)

    handler = ModelSectionHandler(context.asset_store, model_handlers)

    registry.register(ModelSection, handler)

    # Optimizer
    optimizer_handlers = context.get_registry(OptimizerHandler)

    handler = OptimizerSectionHandler(optimizer_handlers)

    registry.register(OptimizerSection, handler)

    # Preference Optimization Criterion
    po_finetune_unit_handlers = context.get_registry(POFinetuneUnitHandler)

    handler = POCriterionSectionHandler(po_finetune_unit_handlers)

    registry.register(POCriterionSection, handler)

    # Sampler
    sampler_handlers = context.get_registry(SamplerHandler)

    handler = SamplerSectionHandler(sampler_handlers)

    registry.register(SamplerSection, handler)

    # Sequence Generator
    seq_generator_handlers = context.get_registry(SequenceGeneratorHandler)

    handler = SequenceGeneratorSectionHandler(seq_generator_handlers)

    registry.register(SequenceGeneratorSection, handler)

    # Sequence-to-Sequence Generator
    seq2seq_generator_handlers = context.get_registry(Seq2SeqGeneratorHandler)

    handler = Seq2SeqGeneratorSectionHandler(seq2seq_generator_handlers)

    registry.register(Seq2SeqGeneratorSection, handler)
