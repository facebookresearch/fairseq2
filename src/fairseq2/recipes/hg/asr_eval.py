# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import torch
from datasets import (  # type: ignore[attr-defined,import-untyped,import-not-found]
    Dataset,
    load_dataset,
)

from fairseq2.config_registry import ConfigRegistry
from fairseq2.data.data_pipeline import SequenceData
from fairseq2.data.text import TextTokenizer, load_text_tokenizer
from fairseq2.datasets.batching import StaticBatching
from fairseq2.logging import get_log_writer
from fairseq2.models.seq2seq import Seq2SeqBatch
from fairseq2.models.sequence import SequenceBatch
from fairseq2.models.wav2vec2.asr import load_wav2vec2_asr_model
from fairseq2.nn.padding import get_seqs_and_padding_mask
from fairseq2.recipes.hg.dataset import Example, create_hf_reader
from fairseq2.recipes.hg.evaluator import HFEvaluator
from fairseq2.recipes.utils.setup import setup_root_gang
from fairseq2.typing import META, DataType
from fairseq2.utils.profiler import Stopwatch

log = get_log_writer(__name__)


@dataclass(kw_only=True)
class AsrEvalConfig:
    """Holds the configuration of a ASR evaluation recipe."""

    # Data
    dataset_name: str
    """The HF dataset to evaluate with."""

    # Model
    model_name: str
    """The name of the model to evaluate."""

    # converter: Callable[[Example], Seq2SeqBatch]
    # """The converter function to convert collated data into Seq2SeqBatch"""

    tokenizer_name: str = "librispeech_asr"
    """The tokenizer to use."""

    split: str = "test"
    """The name of the dataset split to evaluate with."""

    min_audio_len: int = 1
    """The minimum audio sequence length."""

    max_audio_len: int = 800_000
    """The maximum audio sequence length."""

    max_num_elements: int = 3_200_000
    """The maximum number of elements per batch."""

    normalize_audio: bool = False
    """If ``True``, normalizes audio to have zero mean and unit variance."""

    max_samples: int | None = None
    """Maximum number of samples from the dataset to be evaluated. Used
    e.g. for debugging. Default is None, meaning all samples will be evaluated"""

    num_prefetch: int = 4
    """The number of batches to prefetch in background."""

    checkpoint_dir: Path | None = None
    """The checkpoint directory containing models saved by a :class:`FileCheckpointManager`."""

    dtype: DataType = torch.float16
    """The data type of the model."""


asr_eval_presets = ConfigRegistry[AsrEvalConfig]()

asr_eval_preset = asr_eval_presets.decorator


@asr_eval_preset("librispeech_asr")
def _librispeech_asr_config() -> AsrEvalConfig:
    return AsrEvalConfig(
        dataset_name="librispeech_asr",
        model_name="wav2vec2_asr_base_10h",
        split="test.other",
        # converter=librispeech_asr_to_batch,
    )


def _librispeech_asr_to_batch(examples: Example) -> Seq2SeqBatch:
    """
    Converts a collated batch of examples into a Seq2SeqBatch.

    Args:
        examples (dict): A dictionary containing "audio" and "text" keys.

    Returns:
        Seq2SeqBatch: A batch of audio and text sequences.
    """
    source_data = cast(SequenceData, examples["audio"])
    target_data = cast(SequenceData, examples["text"])

    source_seqs, source_padding_mask = get_seqs_and_padding_mask(source_data)
    target_seqs, target_padding_mask = get_seqs_and_padding_mask(target_data)

    return Seq2SeqBatch(
        source_seqs,
        source_padding_mask,
        target_seqs,
        target_padding_mask,
        examples,
    )


@lru_cache(maxsize=None)
def get_cached_tokenizer(tokenizer_name: str) -> TextTokenizer:
    return load_text_tokenizer(tokenizer_name)


def _preprocess_example(
    example: Example, tokenizer_name: str, device: torch.device
) -> Example:
    """
    Preprocesses an individual example by converting the audio array to a PyTorch tensor
    and encoding the text.

    Args:
        example (dict): A dictionary containing "audio" and "text" keys.
        tokenizer_name (str): The name of the tokenizer to use.
        device (torch.device): The device to store the tensors.

    Returns:
        dict: A dictionary with "audio" and "text" as PyTorch tensors.
    """
    tokenizer = get_cached_tokenizer(tokenizer_name)
    encoder = tokenizer.create_encoder(device=device)
    audio_tensor = (
        torch.from_numpy(example["audio"]["array"]).to(torch.float16).to(device)
    )
    text_tensor = encoder(example["text"].lower()).to(device)
    return {"audio": audio_tensor, "text": text_tensor}


def seq2seq_preprocessor(batch: Seq2SeqBatch) -> tuple[SequenceBatch, SequenceBatch]:
    return SequenceBatch(batch.source_seqs, batch.source_padding_mask), SequenceBatch(
        batch.target_seqs, batch.target_padding_mask
    )


def postprocesser(
    outputs: Any, targets: SequenceBatch, tokenizer: TextTokenizer
) -> tuple[list[str], list[str]]:
    decoder = tokenizer.create_decoder()
    pad_idx = tokenizer.vocab_info.pad_idx

    hypotheses, _ = outputs.generate_hypotheses(pad_idx=pad_idx)
    predictions = [decoder(item) for item in hypotheses]
    references = [decoder(item) for item in targets.seqs.to(torch.int32)]

    return predictions, references


def load_wav2vec2_asr_evaluator(
    config: AsrEvalConfig, output_dir: Path
) -> HFEvaluator[Seq2SeqBatch]:
    """
    Load the evaluator used for downstream evaluation of the model
    in a downstream dataset and report BLEU scores

    Args:
        config (HFEvalConfig): The configuration for the evaluation.
        output_dir (Path): The output directory to store the evaluation results.

    Returns:
        HFEvaluator: Evaluation process results.
    """
    if not isinstance(config, AsrEvalConfig):
        raise ValueError(f"Expect AsrEvalConfig, get {type(config)}")

    iterable_ds = load_dataset(config.dataset_name, split=config.split, streaming=True)
    # Load a subset of the dataset if max_samples is set
    ds = Dataset.from_generator(
        lambda: itertools.islice(iterable_ds, 0, config.max_samples),
        features=iterable_ds.features,
    )

    gang = setup_root_gang(log)

    if gang.rank == 0:
        init_device = gang.device
    else:
        init_device = META

    ds = ds.map(lambda x: _preprocess_example(x, config.tokenizer_name, init_device))
    format = {
        "type": "torch",
        "format_kwargs": {"dtype": torch.float16, "device": init_device},
    }
    ds.set_format(**format, columns=["audio", "text"])

    tokenizer = get_cached_tokenizer(config.tokenizer_name)

    pipeline_reader = create_hf_reader(
        dataset=ds,
        gang=gang,
        converter=_librispeech_asr_to_batch,
        batching=StaticBatching(config.max_num_elements),
        num_prefetch=config.num_prefetch,
        pad_value=tokenizer.vocab_info.pad_idx,
        max_seq_len=config.max_audio_len,
    )

    model = load_wav2vec2_asr_model(
        config.model_name, device=init_device, dtype=config.dtype
    )

    wall_watch = Stopwatch(start=True, device=init_device)

    return HFEvaluator[Seq2SeqBatch](
        model=model,
        metrics=["bleu"],
        gang=gang,
        data_reader=pipeline_reader,
        wall_watch=wall_watch,
        preprocessor=seq2seq_preprocessor,
        postprocessor=lambda x, y: postprocesser(x, y, tokenizer),
    )
