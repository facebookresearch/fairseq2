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
from fairseq2.datasets import LengthBatching, SequenceBatch, SyncMode
from fairseq2.datasets.speech import (
    GENERIC_SPEECH_DATASET_FAMILY,
    SpeechDataset,
    SpeechReadOptions,
)
from fairseq2.device import CPU
from fairseq2.metrics import MetricBag
from fairseq2.models.wav2vec2 import Wav2Vec2Model
from fairseq2.recipes import Evaluator, EvalUnit, Model
from fairseq2.recipes.common import (
    create_evaluator,
    load_dataset,
    register_extra_asset_paths,
    setup_gangs,
    setup_reference_model,
    setup_torch,
)
from fairseq2.recipes.config import (
    CommonSection,
    DatasetSection,
    EvaluatorSection,
    GangSection,
    ReferenceModelSection,
)
from fairseq2.utils.rng import manual_seed
from fairseq2.utils.structured import structure
from fairseq2.utils.validation import validate

# isort: split

from fairseq2.recipes.wav2vec2._config import Wav2Vec2LossSection
from fairseq2.recipes.wav2vec2._criterion import Wav2Vec2Criterion


@dataclass(kw_only=True)
class Wav2Vec2EvalConfig:
    model: ReferenceModelSection = field(
        default_factory=lambda: ReferenceModelSection(name="wav2vec2_base")
    )

    dataset: Wav2Vec2EvalDatasetSection = field(
        default_factory=lambda: Wav2Vec2EvalDatasetSection()
    )

    gang: GangSection = field(default_factory=lambda: GangSection())

    evaluator: EvaluatorSection = field(
        default_factory=lambda: EvaluatorSection(dtype=torch.float16)
    )

    loss: Wav2Vec2LossSection = field(default_factory=lambda: Wav2Vec2LossSection())

    common: CommonSection = field(default_factory=lambda: CommonSection())


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

    extras: dict[str, object] = field(default_factory=dict)
    """The dataset-specific extra options."""


def register_wav2vec2_eval_configs(context: RuntimeContext) -> None:
    registry = context.get_config_registry(Wav2Vec2EvalConfig)

    preset = registry.decorator

    @preset("base_ls960h")
    def base_ls960h() -> Wav2Vec2EvalConfig:
        return Wav2Vec2EvalConfig()


@torch.inference_mode()
def load_wav2vec2_evaluator(
    context: RuntimeContext, config: object, output_dir: Path
) -> Evaluator:
    config = structure(config, Wav2Vec2EvalConfig)

    validate(config)

    register_extra_asset_paths(context, config.common.assets)

    setup_torch(context, config.common.torch, output_dir)

    gangs = setup_gangs(context, config.gang)

    seed = config.common.seed

    manual_seed(seed, CPU, gangs.root.device)

    seed += 1

    model = setup_reference_model(
        Wav2Vec2Model,
        context,
        config.model,
        gangs,
        config.evaluator.dtype,
        config.evaluator.amp,
    )

    dataset = load_dataset(SpeechDataset, context, config.dataset, gangs)

    # Initialize the unit.
    criterion = Wav2Vec2Criterion(
        model.module, config.loss.diversity_weight, config.loss.features_penalty_weight
    )

    unit = Wav2Vec2EvalUnit(model, criterion)

    batching = LengthBatching(config.dataset.max_num_elements)

    read_options = SpeechReadOptions(
        batching=batching,
        dtype=config.evaluator.dtype,
        normalize_audio=config.dataset.normalize_audio,
        sync_mode=SyncMode.UNTIL_LAST,
        num_prefetch=config.dataset.num_prefetch,
        seed=seed,
        extras=config.dataset.extras,
    )

    data_reader = dataset.create_reader(
        config.dataset.split,
        gangs.dp,
        config.dataset.min_audio_len,
        config.dataset.max_audio_len,
        read_options,
    )

    units = [unit]

    data_readers = [data_reader]

    seed += 1

    return create_evaluator(
        context,
        config.evaluator,
        config.common,
        output_dir,
        units,
        data_readers,
        gangs,
        seed,
        hyper_params=config,
    )


@final
class Wav2Vec2EvalUnit(EvalUnit[SequenceBatch]):
    _model: Model
    _criterion: Wav2Vec2Criterion

    def __init__(self, model: Model, criterion: Wav2Vec2Criterion) -> None:
        self._model = model

        self._criterion = criterion

    @override
    def __call__(self, batch: SequenceBatch, metric_bag: MetricBag) -> None:
        self._criterion(batch, metric_bag)

    @property
    @override
    def model(self) -> Model:
        return self._model
