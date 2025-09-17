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

from fairseq2.context import RuntimeContext
from fairseq2.datasets import LengthBatching, SyncMode
from fairseq2.datasets.sonarspeech import (
    GENERIC_SONAR_SPEECH_DATASET_FAMILY,
    GenericSonarSpeechDataset,
)
from fairseq2.datasets.speech import (
    GENERIC_SPEECH_DATASET_FAMILY,
    SpeechDataset,
    SpeechReadOptions,
)
from fairseq2.gang import Gangs
from fairseq2.models.seq2seq import Seq2SeqBatch, SonarSpeechSeq2SeqBatch
from fairseq2.models.sequence import SequenceBatch
from fairseq2.models.wav2vec2.sonar import SonarSpeechEncoderModel
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

# isort: split

from fairseq2.recipes.wav2vec2.sonar._criterion import (
    SonarSpeechCriterion,
    SonarSpeechMetricBag,
)
from fairseq2.typing import CPU
from fairseq2.utils.rng import manual_seed
from fairseq2.utils.structured import structure
from fairseq2.utils.validation import validate
from typing_extensions import override


@dataclass(kw_only=True)
class SonarSpeechEvalConfig:
    model: ReferenceModelSection = field(
        default_factory=lambda: ReferenceModelSection(name="wav2vec2_base")
    )

    dataset: SonarSpeechEvalDatasetSection = field(
        default_factory=lambda: SonarSpeechEvalDatasetSection()
    )

    gang: GangSection = field(default_factory=lambda: GangSection())

    evaluator: EvaluatorSection = field(
        default_factory=lambda: EvaluatorSection(dtype=torch.float16)
    )

    common: CommonSection = field(default_factory=lambda: CommonSection())


@dataclass(kw_only=True)
class SonarSpeechEvalDatasetSection(DatasetSection):
    name: str | None = "librispeech_960h"

    family: str = GENERIC_SONAR_SPEECH_DATASET_FAMILY

    path: Path | None = None

    split: str = "valid"

    min_audio_len: int = 1
    """The minimum audio sequence length."""

    max_audio_len: int = 800_000
    """The maximum audio sequence length."""

    max_num_elements: int = 1_280_000
    """The maximum number of elements per batch."""

    normalize_audio: bool = False
    """If ``True``, normalizes audio to have zero mean and unit variance."""

    num_prefetch: int = 4
    """The number of batches to prefetch in background."""

    extras: dict[str, object] = field(default_factory=dict)
    """The dataset-specific extra options."""

    no_padding: bool = False


def register_sonar_speech_eval_configs(context: RuntimeContext) -> None:
    registry = context.get_config_registry(SonarSpeechEvalConfig)

    preset = registry.decorator

    @preset("base_ls960h")
    def base_ls960h() -> SonarSpeechEvalConfig:
        return SonarSpeechEvalConfig()


@torch.inference_mode()
def load_sonar_speech_evaluator(
    context: RuntimeContext, config: object, output_dir: Path
) -> Evaluator[SonarSpeechSeq2SeqBatch]:
    config = structure(config, SonarSpeechEvalConfig)

    validate(config)

    register_extra_asset_paths(context, config.common.assets)

    setup_torch(context, config.common.torch, output_dir)

    gangs = setup_gangs(context, config.gang)

    seed = config.common.seed

    manual_seed(seed, CPU, gangs.root.device)

    seed += 1

    model = setup_reference_model(
        SonarSpeechEncoderModel,
        context,
        config.model,
        gangs,
        config.evaluator.dtype,
        config.evaluator.amp,
        config.evaluator.torch_compile,
    )

    tokenizer = load_text_tokenizer(context, config.tokenizer)

    dataset = load_dataset(GenericSonarSpeechDataset, context, config.dataset, gangs)

    # Initialize the unut.
    criterion = SonarSpeechCriterion(model)

    unit = SonarSpeechEvalUnit(criterion, gangs)

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
        config.dataset.valid_split,
        tokenizer,
        gangs.dp,
        config.dataset.min_audio_len,
        config.dataset.max_audio_len,
        read_options,
    )

    seed += 1

    return create_evaluator(
        context,
        config.evaluator,
        config.common,
        output_dir,
        [unit],
        [data_reader],
        gangs,
        seed,
    )


@final
class SonarSpeechEvalUnit(EvalUnit[SonarSpeechSeq2SeqBatch]):
    _criterion: SonarSpeechCriterion
    _metric_bag: SonarSpeechMetricBag

    def __init__(
        self, criterion: SonarSpeechCriterion, gangs: Gangs, name: None | str = None
    ) -> None:
        self._criterion = criterion

        self._metric_bag = SonarSpeechMetricBag(gangs.dp, train=False)
        self._name = name

    @override
    def __call__(self, batch: SonarSpeechSeq2SeqBatch) -> None:
        self._criterion(batch, self._metric_bag)

    @property
    @override
    def model(self) -> Model:
        return self._criterion.model

    @property
    @override
    def metric_bag(self) -> SonarSpeechMetricBag:
        return self._metric_bag

    @property
    @override
    def name(self) -> None | str:
        return self._name
