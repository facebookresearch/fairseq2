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

from fairseq2.assets import AssetNotFoundError, default_asset_store
from fairseq2.checkpoint import CheckpointModelMetadataProvider
from fairseq2.config_registry import ConfigRegistry
from fairseq2.data.text import load_text_tokenizer
from fairseq2.datasets import LengthBatching
from fairseq2.datasets.asr import GenericAsrDataset, load_asr_dataset
from fairseq2.gang import Gang
from fairseq2.logging import get_log_writer
from fairseq2.models import load_model
from fairseq2.models.seq2seq import Seq2SeqBatch
from fairseq2.models.wav2vec2.asr import Wav2Vec2AsrModel
from fairseq2.nn.utils.module import remove_parametrizations
from fairseq2.recipes.evaluator import AbstractEvalUnit, Evaluator
from fairseq2.recipes.utils.asset import (
    AssetReference,
    asset_as_path,
    retrieve_asset_card,
)
from fairseq2.recipes.utils.log import log_model
from fairseq2.recipes.utils.setup import broadcast_model, setup_root_gang
from fairseq2.recipes.wav2vec2.asr.common import (
    Wav2Vec2AsrCriterion,
    Wav2Vec2AsrMetricBag,
    Wav2Vec2AsrScorer,
)
from fairseq2.typing import META, DataType
from fairseq2.utils.profiler import Stopwatch

log = get_log_writer(__name__)


@dataclass(kw_only=True)
class Wav2Vec2AsrEvalConfig:
    """Holds the configuration of a wav2vec 2.0 ASR model evaluation task."""

    # Data
    dataset: AssetReference = "librilight_asr_10h"
    """The name, path, or path to the asset card of the ASR dataset."""

    split: str = "test_other"
    """The name of the eval data split."""

    min_audio_len: int = 1
    """The minimum audio sequence length."""

    max_audio_len: int = 800_000
    """The maximum audio sequence length."""

    max_num_elements: int = 3_200_000
    """The maximum number of elements per batch."""

    normalize_audio: bool = False
    """If ``True``, normalizes audio to have zero mean and unit variance."""

    num_prefetch: int = 4
    """The number of batches to prefetch in background."""

    # Model
    model: AssetReference = "wav2vec2_asr_base_10h"
    """The name or path to the asset card of the wav2vec 2.0 ASR model to evaluate."""

    checkpoint_dir: Path | None = None
    """The checkpoint directory containing models saved by :class:`FileCheckpointManager`."""

    dtype: DataType = torch.float16
    """The data type of the model."""

    amp: bool = False
    """If ``True``, runs evaluation with ``torch.amp``."""

    # Misc
    seed: int = 2
    """The random number generator seed to use."""


wav2vec2_asr_eval_presets = ConfigRegistry[Wav2Vec2AsrEvalConfig]()

wav2vec2_asr_eval_preset = wav2vec2_asr_eval_presets.decorator


@wav2vec2_asr_eval_preset("base_10h")
def _base_10h() -> Wav2Vec2AsrEvalConfig:
    return Wav2Vec2AsrEvalConfig()


@torch.inference_mode()
def load_wav2vec2_asr_evaluator(
    config: Wav2Vec2AsrEvalConfig, output_dir: Path
) -> Evaluator[Seq2SeqBatch]:
    """Load an :class:`Evaluator` for wav2vec 2.0 ASR model evaluation."""
    wall_watch = Stopwatch(start=True)

    if config.checkpoint_dir is not None:
        default_asset_store.metadata_providers.append(
            CheckpointModelMetadataProvider(config.checkpoint_dir)
        )

    gang = setup_root_gang(log)

    model_card = retrieve_asset_card(config.model)

    # Load the tokenizer.
    log.info("Loading {} tokenizer.", model_card.name)

    tokenizer = load_text_tokenizer(model_card)

    log.info("Tokenizer loaded.")

    # Load the dataset.
    try:
        dataset_card = retrieve_asset_card(config.dataset)
    except AssetNotFoundError:
        dataset_card = None

    if dataset_card is not None:
        log.info("Loading {} ASR dataset.", dataset_card.name)

        dataset = load_asr_dataset(dataset_card)

        log.info("Dataset loaded.")
    else:
        dataset_path = asset_as_path(config.dataset)

        dataset = GenericAsrDataset.from_path(dataset_path)

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

    if not isinstance(model, Wav2Vec2AsrModel):
        raise ValueError(
            f"The model must be of type `{Wav2Vec2AsrModel}`, but is of type `{type(model)}` instead."
        )

    gang.barrier()

    log.info("Model loaded on rank 0.")

    remove_parametrizations(model)

    # Distribute the model to all processes in the gang.
    if gang.size != 1:
        broadcast_model(model, gang, log)

    log_model(model, log)

    # Initialize the criterion.
    ref_output_file = output_dir.joinpath(f"transcriptions/rank_{gang.rank}.ref.txt")
    hyp_output_file = output_dir.joinpath(f"transcriptions/rank_{gang.rank}.hyp.txt")

    try:
        ref_output_file.parent.mkdir(parents=True, exist_ok=True)
    except OSError as ex:
        raise RuntimeError(
            f"The output directory '{ref_output_file.parent}' cannot be created. See nested exception for details."
        ) from ex

    try:
        ref_output_fp = ref_output_file.open("w")
    except OSError as ex:
        raise RuntimeError(
            f"The output file '{ref_output_file}' cannot be created. See nested exception for details."
        ) from ex

    try:
        hyp_output_fp = hyp_output_file.open("w")
    except OSError as ex:
        raise RuntimeError(
            f"The output file '{hyp_output_file}' cannot be created. See nested exception for details."
        ) from ex

    scorer = Wav2Vec2AsrScorer(
        tokenizer, ref_output_stream=ref_output_fp, hyp_output_stream=hyp_output_fp
    )

    criterion = Wav2Vec2AsrCriterion(model, scorer)

    # Initialize the unit.
    unit = Wav2Vec2AsrEvalUnit(criterion, gang)

    seed = config.seed

    try:
        data_reader = dataset.create_reader(
            config.split,
            tokenizer,
            gang,
            batching=LengthBatching(config.max_num_elements),
            dtype=config.dtype,
            min_audio_len=config.min_audio_len,
            max_audio_len=config.max_audio_len,
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

    # Initialize the evaluator.
    return Evaluator[Seq2SeqBatch](
        units=[unit],
        data_readers=[data_reader],
        root_gang=gang,
        dtype=config.dtype,
        amp=config.amp,
        tb_dir=output_dir.joinpath("tb"),
        metrics_dir=output_dir.joinpath("metrics"),
        seed=seed,
        wall_watch=wall_watch,
    )


@final
class Wav2Vec2AsrEvalUnit(AbstractEvalUnit[Seq2SeqBatch]):
    _criterion: Wav2Vec2AsrCriterion
    _metric_bag: Wav2Vec2AsrMetricBag

    def __init__(self, criterion: Wav2Vec2AsrCriterion, gang: Gang) -> None:
        super().__init__(criterion.model)

        self._criterion = criterion

        self._metric_bag = Wav2Vec2AsrMetricBag(gang, train=False)

    @override
    def __call__(self, batch: Seq2SeqBatch) -> None:
        self._criterion(batch, self._metric_bag)

    @property
    @override
    def metric_bag(self) -> Wav2Vec2AsrMetricBag:
        return self._metric_bag
