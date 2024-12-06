# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import final

import torch
from typing_extensions import override

from fairseq2.assets import AssetNotFoundError
from fairseq2.config_registry import ConfigRegistry
from fairseq2.datasets.batching import LengthBatching
from fairseq2.datasets.speech import GenericSpeechDataset, load_speech_dataset
from fairseq2.dependency import resolve, resolve_all
from fairseq2.gang import Gang
from fairseq2.logging import get_log_writer
from fairseq2.metrics import MetricRecorder
from fairseq2.models import load_model
from fairseq2.models.sequence import SequenceBatch
from fairseq2.models.bestrq import BestRQModel
from fairseq2.nn.utils.module import remove_parametrizations
from fairseq2.recipes.evaluator import AbstractEvalUnit, Evaluator
from fairseq2.recipes.utils.asset import (
    AssetReference,
    asset_as_path,
    retrieve_asset_card,
)
from fairseq2.recipes.utils.log import log_model
from fairseq2.recipes.utils.setup import broadcast_model
from fairseq2.recipes.bestrq.common import BestRQCriterion, BestRQMetricBag
from fairseq2.typing import META, DataType
from fairseq2.utils.profiler import Stopwatch

log = get_log_writer(__name__)


@dataclass(kw_only=True)
class BestRQEvalConfig:
    """Holds the configuration of a wav2vec 2.0 model evaluation task."""

    # Data
    dataset: AssetReference = "librispeech_960h"
    """The name, path or path to the asset card of the dataset to evaluate on."""

    split: str = "valid"
    """The name of the eval data split."""

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

    # Model
    model: AssetReference = "bestrq_base"
    """The name or path to the asset card of the wav2vec 2.0 model to evaluate."""

    checkpoint_dir: Path | None = None
    """The checkpoint directory containing models saved by :class:`FileCheckpointManager`."""

    dtype: DataType = torch.float16
    """The data type of the model."""

    amp: bool = False
    """If ``True``, runs evaluation with ``torch.amp``."""

    # Loss
    label_smoothing: float = 1
    """Value for label smoothing."""

    # Score
    score_metric: str | None = "loss"
    """The name of the metric to use as score."""

    lower_score_better: bool = True
    """The ``True``, lower scores are considered better."""

    # Misc
    seed: int = 2
    """The random number generator seed to use."""


bestrq_eval_presets = ConfigRegistry[BestRQEvalConfig]()

bestrq_eval_preset = bestrq_eval_presets.decorator


@bestrq_eval_preset("base_ls960h")
def _base_ls960h() -> BestRQEvalConfig:
    return BestRQEvalConfig()


@torch.inference_mode()
def load_bestrq_evaluator(
    config: BestRQEvalConfig, output_dir: Path
) -> Evaluator[SequenceBatch]:
    """Load an :class:`Evaluator` for wav2vec 2.0 model evaluation."""
    wall_watch = Stopwatch(start=True)

    gang = resolve(Gang)

    # Load the dataset.
    try:
        dataset_card = retrieve_asset_card(config.dataset)
    except AssetNotFoundError:
        dataset_card = None

    if dataset_card is not None:
        log.info("Loading {} speech dataset.", dataset_card.name)

        dataset = load_speech_dataset(dataset_card)

        log.info("Dataset loaded.")
    else:
        dataset_path = asset_as_path(config.dataset)

        dataset = GenericSpeechDataset.from_path(dataset_path)

    model_card = retrieve_asset_card(config.model)

    # Load the model.
    log.info("Loading {} model on rank 0.", model_card.name)

    if gang.rank == 0:
        init_device = gang.device
    else:
        init_device = META

    try:
        model = load_model(model_card, device=init_device, dtype=config.dtype)
    except ValueError as ex:
        raise ValueError(
            "The model cannot be initialized. See nested exception for details."
        ) from ex

    if not isinstance(model, BestRQModel):
        raise ValueError(
            f"The model must be of type `{BestRQModel}`, but is of type `{type(model)}` instead."
        )

    gang.barrier()

    log.info("Model loaded on rank 0.")

    remove_parametrizations(model)

    # Distribute the model to all processes in the gang.
    if gang.size != 1:
        broadcast_model(model, gang, log)

    log_model(model, log)

    # Initialize the criterion.
    criterion = BestRQCriterion(
        model, config.diversity_loss_weight, config.feature_penalty_weight
    )

    # Initialize the unit.
    unit = BestRQEvalUnit(criterion, gang)

    seed = config.seed

    try:
        data_reader = dataset.create_reader(
            config.split,
            gang,
            config.max_audio_len,
            batching=LengthBatching(config.max_num_elements),
            dtype=config.dtype,
            min_audio_len=config.min_audio_len,
            normalize_audio=config.normalize_audio,
            sync_mode="until_last",
            num_prefetch=config.num_prefetch,
            seed=seed,
        )
    except ValueError as ex:
        raise ValueError(
            "The data reader cannot be initialized. See nested exception for details."
        ) from ex

    seed += 1

    metric_recorders = resolve_all(MetricRecorder)

    # Initialize the evaluator.
    return Evaluator[SequenceBatch](
        units=[unit],
        data_readers=[data_reader],
        root_gang=gang,
        dtype=config.dtype,
        amp=config.amp,
        metric_recorders=metric_recorders,
        seed=seed,
        wall_watch=wall_watch,
    )


@final
class BestRQEvalUnit(AbstractEvalUnit[SequenceBatch]):
    _criterion: BestRQCriterion
    _metric_bag: BestRQMetricBag

    def __init__(self, criterion: BestRQCriterion, gang: Gang) -> None:
        super().__init__(criterion.model)

        self._criterion = criterion

        self._metric_bag = BestRQMetricBag(gang, train=False)

    @override
    def __call__(self, batch: SequenceBatch) -> None:
        self._criterion(batch, self._metric_bag)

    @property
    @override
    def metric_bag(self) -> BestRQMetricBag:
        return self._metric_bag
