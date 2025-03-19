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
from fairseq2.models.sequence import SequenceBatch
from fairseq2.models.wav2vec2 import Wav2Vec2Model
from fairseq2.recipes import Evaluator, EvalUnit, Model
from fairseq2.recipes.common import (
    create_evaluator,
    load_dataset,
    register_extra_asset_paths,
    setup_gangs,
    setup_reference_model,
)
from fairseq2.recipes.config import (
    CommonSection,
    DatasetSection,
    EvaluatorSection,
    GangSection,
    ReferenceModelSection,
)
from fairseq2.recipes.wav2vec2._common import (
    Wav2Vec2Criterion,
    Wav2Vec2LossSection,
    Wav2Vec2MetricBag,
)
from fairseq2.typing import CPU
from fairseq2.utils.rng import manual_seed
from fairseq2.utils.structured import structure
from fairseq2.utils.validation import validate


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
) -> Evaluator[SequenceBatch]:
    config = structure(config, Wav2Vec2EvalConfig)

    validate(config)

    register_extra_asset_paths(context, config)

    torch.set_float32_matmul_precision("high")

    gangs = setup_gangs(context, config)

    seed = config.common.seed

    manual_seed(seed, CPU, gangs.root.device)

    seed += 1

    model = setup_reference_model(
        Wav2Vec2Model,
        context,
        config.model.name,
        gangs,
        config.evaluator.dtype,
        config.evaluator.amp,
        config.evaluator.torch_compile,
    )

    dataset = load_dataset(SpeechDataset, context, config, gangs)

    # Initialize the unut.
    criterion = Wav2Vec2Criterion(
        model, config.loss.diversity_loss_weight, config.loss.feature_penalty_weight
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
        extras=config.dataset.extras,
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
class Wav2Vec2EvalUnit(EvalUnit[SequenceBatch]):
    _criterion: Wav2Vec2Criterion
    _metric_bag: Wav2Vec2MetricBag

    def __init__(self, criterion: Wav2Vec2Criterion, gangs: Gangs) -> None:
        self._criterion = criterion

        self._metric_bag = Wav2Vec2MetricBag(gangs.dp, train=False)

    @override
    def __call__(self, batch: SequenceBatch) -> None:
        self._criterion(batch, self._metric_bag)

    @property
    @override
    def model(self) -> Model:
        return self._criterion.model

    @property
    @override
    def metric_bag(self) -> Wav2Vec2MetricBag:
        return self._metric_bag
