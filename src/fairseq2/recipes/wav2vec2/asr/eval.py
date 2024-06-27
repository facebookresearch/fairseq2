# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import torch

from fairseq2.assets import default_asset_store
from fairseq2.assets.utils import retrieve_asset_card
from fairseq2.checkpoint import CheckpointModelMetadataProvider
from fairseq2.config_registry import ConfigRegistry
from fairseq2.data.text import load_text_tokenizer
from fairseq2.datasets.asr import load_asr_dataset
from fairseq2.logging import get_log_writer
from fairseq2.models.seq2seq import Seq2SeqBatch
from fairseq2.models.wav2vec2.asr import load_wav2vec2_asr_model
from fairseq2.nn.utils.module import remove_parametrizations
from fairseq2.recipes.evaluator import Evaluator
from fairseq2.recipes.utils.log import log_model
from fairseq2.recipes.utils.setup import broadcast_model, setup_root_gang
from fairseq2.recipes.wav2vec2.asr.units import Wav2Vec2AsrEvalUnit
from fairseq2.typing import META, DataType
from fairseq2.utils.profiler import Stopwatch

log = get_log_writer(__name__)


@dataclass
class Wav2Vec2AsrEvalConfig:
    """Holds the evaluation configuration of a wav2vec 2.0 ASR model."""

    # Data
    dataset: Union[str, Path] = "librilight_asr_10h"
    """The name or path to the asset card of the dataset to evaluate with."""

    split: str = "test_other"
    """The name of the dataset split to evaluate with."""

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
    model: Union[str, Path] = "wav2vec2_asr_base_10h"
    """The name or path to the asset card of the wav2vec 2.0 ASR model to evaluate."""

    checkpoint_dir: Optional[Path] = None
    """The checkpoint directory containing models saved by :class:`FileCheckpointManager`."""

    dtype: DataType = torch.float16
    """The data type of the model."""

    # Misc
    seed: int = 2
    """The random number generator seed to use."""


wav2vec2_asr_eval_presets = ConfigRegistry[Wav2Vec2AsrEvalConfig]()

wav2vec2_asr_eval_preset = wav2vec2_asr_eval_presets.decorator


def _base_10h() -> Wav2Vec2AsrEvalConfig:
    return Wav2Vec2AsrEvalConfig()


def _register_eval() -> None:
    wav2vec2_asr_eval_presets.register("base_10h", _base_10h)


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

    # Load the tokenizer.
    model_card = retrieve_asset_card(config.model)

    log.info("Loading {} tokenizer.", model_card.name)

    tokenizer = load_text_tokenizer(model_card)

    log.info("Tokenizer loaded.")

    # Load the dataset.
    dataset_card = retrieve_asset_card(config.dataset)

    log.info("Loading {} ASR dataset.", dataset_card.name)

    dataset = load_asr_dataset(dataset_card)

    log.info("Dataset loaded.")

    # Load the model.
    log.info("Loading {} model on rank 0.", model_card.name)

    if gang.rank == 0:
        init_device = gang.device
    else:
        init_device = META

    model = load_wav2vec2_asr_model(model_card, device=init_device, dtype=config.dtype)

    gang.barrier()

    log.info("Model loaded on rank 0.")

    remove_parametrizations(model)

    # Distribute the model to all processes in the gang.
    if gang.size != 1:
        broadcast_model(model, gang, log)

    log_model(model, log)

    # Initialize the evaluation unit.
    wer_file = output_dir.joinpath(f"wer/rank_{gang.rank}.txt")

    try:
        wer_file.parent.mkdir(parents=True, exist_ok=True)
    except OSError as ex:
        raise RuntimeError(
            f"The WER output directory ({wer_file.parent}) cannot be created. See nested exception for details."
        ) from ex

    try:
        wer_fp = wer_file.open("w")
    except OSError as ex:
        raise RuntimeError(
            f"The WER output file ({wer_file}) cannot be created. See nested exception for details."
        ) from ex

    unit = Wav2Vec2AsrEvalUnit(model, gang, tokenizer, output_stream=wer_fp)

    data_reader = dataset.create_reader(
        config.split,
        tokenizer,
        gang,
        dtype=config.dtype,
        min_audio_len=config.min_audio_len,
        max_audio_len=config.max_audio_len,
        max_num_elements=config.max_num_elements,
        normalize_audio=config.normalize_audio,
        num_prefetch=config.num_prefetch,
    )

    # Initialize the evaluator.
    return Evaluator[Seq2SeqBatch](
        units=[unit],
        data_readers=[data_reader],
        root_gang=gang,
        seed=config.seed,
        wall_watch=wall_watch,
    )
