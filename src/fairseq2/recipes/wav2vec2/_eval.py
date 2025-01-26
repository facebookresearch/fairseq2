# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import final

import torch
from typing_extensions import override

from fairseq2.context import RuntimeContext
from fairseq2.datasets import LengthBatching, SyncMode
from fairseq2.datasets.speech import (
    GENERIC_SPEECH_DATASET_FAMILY,
    SpeechDataset,
    SpeechReadOptions,
)
from fairseq2.gang import Gangs
from fairseq2.logging import log
from fairseq2.models.sequence import SequenceBatch
from fairseq2.models.wav2vec2 import Wav2Vec2Model
from fairseq2.nn.utils.module import remove_parametrizations
from fairseq2.recipes.common import (
    broadcast_model,
    compile_eval_model,
    create_evaluator,
    load_dataset,
    load_eval_model,
    register_extra_asset_paths,
    setup_gangs,
)
from fairseq2.recipes.config import DatasetSection, EvalRecipeConfig, EvaluatorSection
from fairseq2.recipes.evaluator import AbstractEvalUnit, Evaluator
from fairseq2.recipes.utils.log import log_model
from fairseq2.recipes.wav2vec2._common import Wav2Vec2Criterion, Wav2Vec2MetricBag
from fairseq2.typing import CPU
from fairseq2.utils.config import process_config
from fairseq2.utils.rng import manual_seed


@dataclass(kw_only=True)
class Wav2Vec2EvalConfig(EvalRecipeConfig):
    """Holds the configuration of a wav2vec 2.0 model evaluation task."""

    model: str = "wav2vec2_base"

    dataset: Wav2Vec2EvalDatasetSection = field(
        default_factory=lambda: Wav2Vec2EvalDatasetSection()
    )

    evaluator: Wav2Vec2EvaluatorSection = field(
        default_factory=lambda: Wav2Vec2EvaluatorSection(dtype=torch.float16)
    )


@dataclass(kw_only=True)
class Wav2Vec2EvalDatasetSection(DatasetSection):
    name: str | None = "librispeech_960h"

    family: str = GENERIC_SPEECH_DATASET_FAMILY

    path: Path | None = None

    split: str = "valid"

    min_audio_len: int = 32_000
    """The minimum audio sequence length."""

    max_audio_len: int = 250_000
    """The maximum audio sequence length."""

    max_num_elements: int = 1_500_000
    """The maximum number of elements per batch."""

    normalize_audio: bool = False
    """If ``True``, normalizes audio to have zero mean and unit variance."""

    num_prefetch: int = 4
    """The number of batches to prefetch in background."""


@dataclass(kw_only=True)
class Wav2Vec2EvaluatorSection(EvaluatorSection):
    diversity_loss_weight: float = 0.1
    """The weight of the diversity loss."""

    feature_penalty_weight: float = 10.0
    """The weight of the regularization penalty applied to the extracted features."""


def register_wav2vec2_eval_configs(context: RuntimeContext) -> None:
    registry = context.get_config_registry(Wav2Vec2EvalConfig)

    preset = registry.decorator

    @preset("base_ls960h")
    def base_ls960h() -> Wav2Vec2EvalConfig:
        return Wav2Vec2EvalConfig()


@torch.inference_mode()
def load_wav2vec2_evaluator(
    context: RuntimeContext, config: Wav2Vec2EvalConfig, output_dir: Path
) -> Evaluator[SequenceBatch]:
    register_extra_asset_paths(context, config.assets)

    process_config(context, config)

    gangs = setup_gangs(context, config.gang)

    dataset = load_dataset(SpeechDataset, context, config.dataset, gangs)

    seed = config.seed

    manual_seed(seed, CPU, context.device)

    seed += 1

    model = load_eval_model(
        Wav2Vec2Model,
        context,
        config.model,
        gangs,
        config.evaluator.dtype,
        mixed_precision=config.evaluator.amp,
    )

    broadcast_model(config.model, model, gangs)

    remove_parametrizations(model)

    log_model(log, model, gangs)

    if config.evaluator.torch_compile:
        model = compile_eval_model(context, config.model, model)

    # Initialize the unut.
    criterion = Wav2Vec2Criterion(
        model,
        config.evaluator.diversity_loss_weight,
        config.evaluator.feature_penalty_weight,
    )

    unit = Wav2Vec2EvalUnit(criterion, gangs)

    batching = LengthBatching(config.dataset.max_num_elements)

    read_options = SpeechReadOptions(
        batching=batching,
        dtype=config.evaluator.dtype,
        normalize_audio=config.dataset.normalize_audio,
        sync_mode=SyncMode.UNTIL_LAST,
        num_prefetch=config.dataset.num_prefetch,
        seed=seed,
    )

    data_reader = dataset.create_reader(
        config.dataset.split,
        gangs.dp,
        config.dataset.min_audio_len,
        config.dataset.max_audio_len,
        read_options,
    )

    seed += 1

    return create_evaluator(
        context, config, output_dir, [unit], [data_reader], gangs, seed
    )


@final
class Wav2Vec2EvalUnit(AbstractEvalUnit[SequenceBatch]):
    _criterion: Wav2Vec2Criterion
    _metric_bag: Wav2Vec2MetricBag

    def __init__(self, criterion: Wav2Vec2Criterion, gangs: Gangs) -> None:
        super().__init__(criterion.model)

        self._criterion = criterion

        self._metric_bag = Wav2Vec2MetricBag(gangs.dp, train=False)

    @override
    def __call__(self, batch: SequenceBatch) -> None:
        self._criterion(batch, self._metric_bag)

    @property
    @override
    def metric_bag(self) -> Wav2Vec2MetricBag:
        return self._metric_bag
