# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import torch
from torch.nn import Module

from fairseq2.assets import default_asset_store
from fairseq2.config_registry import ConfigRegistry
from fairseq2.data.text import load_text_tokenizer
from fairseq2.datasets.asr import load_asr_dataset
from fairseq2.logging import get_log_writer
from fairseq2.models.fsdp import get_fsdp_wrap_policy
from fairseq2.models.seq2seq import Seq2SeqBatch
from fairseq2.models.wav2vec2.asr import load_wav2vec2_asr_model
from fairseq2.nn.ddp import to_ddp
from fairseq2.nn.fsdp import to_fsdp
from fairseq2.nn.utils.module import remove_parametrizations
from fairseq2.recipes.evaluator import StandardEvaluator
from fairseq2.recipes.utils.log import log_model
from fairseq2.recipes.utils.setup import setup_root_gang
from fairseq2.recipes.wav2vec2.asr.criterion import Wav2Vec2AsrCriterion
from fairseq2.typing import META, DataType
from fairseq2.utils.profiler import Stopwatch

log = get_log_writer(__name__)


@dataclass
class Wav2Vec2AsrEvalConfig:
    """Holds the configuration of a wav2vec 2.0 ASR evaluation recipe."""

    # Data
    dataset_name: str = "librilight_asr_10h"
    """The dataset to evaluate with."""

    tokenizer_name: str = "librispeech_asr"
    """The tokenizer to use."""

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
    model_name: str = "wav2vec2_asr_base_10h"
    """The name of the wav2vec 2.0 ASR model to evaluate."""

    checkpoint_dir: Optional[Path] = None
    """The checkpoint directory containing models saved by a :class:`FileCheckpointManager`."""

    dtype: DataType = torch.float16
    """The data type of the model."""

    data_parallelism: Literal["ddp", "fsdp"] = "ddp"
    """The data parallelism API to use."""

    fsdp_wrap_granularity: Literal["layer", "stack", "model"] = "stack"
    """The granularity at which to wrap the ASR model."""


wav2vec2_asr_eval_presets = ConfigRegistry[Wav2Vec2AsrEvalConfig]()

wav2vec2_asr_eval_preset = wav2vec2_asr_eval_presets.decorator


@wav2vec2_asr_eval_preset("base_10h")
def _base_10h() -> Wav2Vec2AsrEvalConfig:
    return Wav2Vec2AsrEvalConfig()


def load_wav2vec2_asr_evaluator(
    config: Wav2Vec2AsrEvalConfig, output_dir: Path
) -> StandardEvaluator[Seq2SeqBatch]:
    """Load a wav2vec 2.0 ASR evaluator."""
    wall_watch = Stopwatch(start=True)

    gang = setup_root_gang(log)

    log.info("Loading {} tokenizer.", config.tokenizer_name)

    tokenizer = load_text_tokenizer(config.tokenizer_name)

    log.info("Tokenizer loaded.")

    log.info("Loading {} dataset.", config.dataset_name)

    dataset = load_asr_dataset(config.dataset_name)

    data_reader = dataset.create_reader(
        split=config.split,
        tokenizer=tokenizer,
        gang=gang,
        dtype=config.dtype,
        min_audio_len=config.min_audio_len,
        max_audio_len=config.max_audio_len,
        max_num_elements=config.max_num_elements,
        normalize_audio=config.normalize_audio,
        num_prefetch=config.num_prefetch,
    )

    log.info("Dataset loaded.")

    if config.checkpoint_dir is not None:
        default_asset_store.add_file_metadata_provider(config.checkpoint_dir)

    log.info("Loading {} model on rank 0.", config.model_name)

    if gang.rank == 0:
        init_device = gang.device
    else:
        init_device = META

    model = load_wav2vec2_asr_model(
        config.model_name, device=init_device, dtype=config.dtype
    )

    log.info("Model loaded on rank 0.")

    gang.barrier()

    # No need for weight normalization outside training.
    remove_parametrizations(model)

    dp_model: Module

    # Set up data parallelism.
    if gang.size == 1:
        dp_model = model
    else:
        log.info("Wrapping the model with {} and broadcasting to all ranks from rank 0.", config.data_parallelism.upper())  # fmt: skip

        if config.data_parallelism == "ddp":
            dp_model = to_ddp(model, gang)
        elif config.data_parallelism == "fsdp":
            wrap_policy, ignored_modules = get_fsdp_wrap_policy(
                model, wrap_granularity=config.fsdp_wrap_granularity
            )

            dp_model = to_fsdp(
                model,
                gang,
                wrap_policy,
                ignored_modules=ignored_modules,
                skip_init=True,
                broadcast_state=True,
            )
        else:
            raise ValueError(
                f"`config.data_parallelism` must be 'ddp' or 'fsdp', but is '{config.data_parallelism}' instead."
            )

        log.info("Model wrapped and broadcasted to all ranks.")

    log_model(dp_model, log)

    # Initialize the criterion.
    wer_file = output_dir.joinpath(f"wer/rank_{gang.rank}.txt")

    try:
        wer_file.parent.mkdir(parents=True, exist_ok=True)
    except OSError as ex:
        raise RuntimeError(
            f"The WER output directory ({wer_file.parent}) cannot be created. See nested exception for details."
        ) from ex

    criterion = Wav2Vec2AsrCriterion(dp_model, gang, tokenizer, wer_file=wer_file)

    # Initialize the evaluator.
    return StandardEvaluator[Seq2SeqBatch](
        criterion=criterion,
        gang=gang,
        data_reader=data_reader,
        wall_watch=wall_watch,
    )
