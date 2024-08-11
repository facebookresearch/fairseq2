# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, cast

import torch
from datasets import (  # type: ignore[attr-defined,import-untyped,import-not-found]
    Dataset,
    load_dataset,
    load_dataset_builder,
)
from transformers import (  # type: ignore[attr-defined,import-untyped,import-not-found]
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from fairseq2.assets.metadata_provider import AssetNotFoundError
from fairseq2.config_registry import ConfigRegistry
from fairseq2.data.data_pipeline import SequenceData
from fairseq2.data.text import load_text_tokenizer
from fairseq2.data.text.text_tokenizer import TextTokenizer
from fairseq2.datasets.batching import StaticBatching
from fairseq2.logging import get_log_writer
from fairseq2.models.model import Model
from fairseq2.models.seq2seq import Seq2SeqBatch
from fairseq2.models.sequence import SequenceBatch
from fairseq2.models.wav2vec2.asr import load_wav2vec2_asr_model
from fairseq2.nn.padding import get_seqs_and_padding_mask
from fairseq2.recipes.hg.dataset import Example, create_hf_reader
from fairseq2.recipes.hg.evaluator import HFEvaluator
from fairseq2.recipes.utils.asset import retrieve_asset_card
from fairseq2.recipes.utils.setup import setup_root_gang
from fairseq2.typing import META, DataType, Device
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


@asr_eval_preset("default_asr")
def _default_asr_config() -> AsrEvalConfig:
    return AsrEvalConfig(
        dataset_name="librispeech_asr",
        model_name="wav2vec2_asr_base_10h",
        split="test.other",
    )


def extract_features(example: Example) -> Example:
    """
    Extracts source and target features from dataset examples.

    Args:
        example (dict): A dictionary containing source and target keys.

    Returns:
        dict: A dictionary containing the extracted features.
    """
    return {"audio": example["audio"]["array"], "text": example["text"].lower()}


def to_batch(examples: Example, model_type: str, device: Device) -> Seq2SeqBatch:
    """
    Converts a collated batch of examples into a Seq2SeqBatch.

    Args:
        examples (dict): A dictionary containing "audio" and "text" keys.

    Returns:
        Seq2SeqBatch: A batch of audio and text sequences.
    """
    source_data = cast(SequenceData, examples["audio"])
    target_data = cast(torch.Tensor, examples["text"])

    if model_type == "wav2vec2":
        source_seqs, source_padding_mask = get_seqs_and_padding_mask(source_data)
        source_seqs = source_seqs.to(device)
        source_padding_mask = (
            source_padding_mask.to(device) if source_padding_mask is not None else None
        )
    elif model_type == "whisper":
        source_seqs = cast(torch.Tensor, source_data).to(device)
        source_padding_mask = None
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    target_seqs = target_data
    target_padding_mask = None

    return Seq2SeqBatch(
        source_seqs,
        source_padding_mask,
        target_seqs,
        target_padding_mask,
        examples,
    )


def prepare_dataset(
    config: AsrEvalConfig, processor: Optional[Callable[[Example], Example]] = None
) -> Dataset:
    """
    Prepares a dataset for evaluation. The dataset is loaded from the
    HF datasets and preprocessed using the provided processor.

    Args:
        config (AsrEvalConfig): The configuration for the evaluation.
        processor (Callable): A function to preprocess examples.

    Returns:
        Dataset: The prepared dataset.
    """
    iterable_ds = load_dataset(config.dataset_name, split=config.split, streaming=True)
    ds = Dataset.from_generator(
        lambda: itertools.islice(iterable_ds, 0, config.max_samples),
        features=iterable_ds.features,
    )
    ds = ds.map(lambda x: extract_features(x))

    if processor is not None:
        ds = ds.map(processor)

    format = {
        "type": "torch",
        "format_kwargs": {"dtype": config.dtype},
    }
    ds.set_format(**format, columns=["audio", "text"])

    return ds


def evaluator_preprocessor(batch: Seq2SeqBatch) -> tuple[SequenceBatch, SequenceBatch]:
    return SequenceBatch(batch.source_seqs, batch.source_padding_mask), SequenceBatch(
        batch.target_seqs, batch.target_padding_mask
    )


def evaluator_postprocesser(
    outputs: Any, targets: SequenceBatch, tokenizer: TextTokenizer
) -> tuple[list[str], list[str]]:
    decoder = tokenizer.create_decoder()
    pad_idx = tokenizer.vocab_info.pad_idx

    hypotheses, _ = outputs.generate_hypotheses(pad_idx=pad_idx)
    predictions = [decoder(item) for item in hypotheses]
    references = cast(list[str], targets.seqs)

    return predictions, references


def load_asr_evaluator(
    config: AsrEvalConfig, output_dir: Path
) -> HFEvaluator[Seq2SeqBatch]:
    """
    Load the evaluator based on the model type.

    Args:
        config (AsrEvalConfig): The configuration for the evaluation.
        output_dir (Path): The output directory to store the evaluation results.

    Returns:
        HFEvaluator: Evaluation process.
    """
    try:
        retrieve_asset_card(config.model_name)
        return load_wav2vec2_asr_evaluator(config, output_dir)
    except AssetNotFoundError:
        return load_hg_asr_evaluator(config, output_dir)


def load_wav2vec2_asr_evaluator(
    config: AsrEvalConfig, output_dir: Path
) -> HFEvaluator[Seq2SeqBatch]:
    """
    Load the evaluator used for downstream evaluation of the wav2vec2 model.

    Args:
        config (HFEvalConfig): The configuration for the evaluation.
        output_dir (Path): The output directory to store the evaluation results.

    Returns:
        HFEvaluator: Evaluation process.
    """
    if not isinstance(config, AsrEvalConfig):
        raise ValueError(f"Expect AsrEvalConfig, get {type(config)}")

    ds = prepare_dataset(config)

    gang = setup_root_gang(log)

    if gang.rank == 0:
        init_device = gang.device
    else:
        init_device = META

    tokenizer = load_text_tokenizer(config.tokenizer_name)

    pipeline_reader = create_hf_reader(
        dataset=ds,
        gang=gang,
        converter=lambda x: to_batch(x, "wav2vec2", init_device),
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
        preprocessor=evaluator_preprocessor,
        postprocessor=lambda x, y: evaluator_postprocesser(x, y, tokenizer),
    )


class HGModelWrapper:
    def __init__(self, model: WhisperForConditionalGeneration):
        self.model = model

    def __call__(self, batch: SequenceBatch) -> Any:
        return self.model.generate(batch.seqs)


def load_hg_asr_evaluator(
    config: AsrEvalConfig, output_dir: Path
) -> HFEvaluator[Seq2SeqBatch]:
    """
    Load the evaluator used for downstream evaluation of the whisper model

    Args:
        config (HFEvalConfig): The configuration for the evaluation.
        output_dir (Path): The output directory to store the evaluation results.

    Returns:
        HFEvaluator: Evaluation process.
    """
    if not isinstance(config, AsrEvalConfig):
        raise ValueError(f"Expect AsrEvalConfig, get {type(config)}")

    gang = setup_root_gang(log)

    if gang.rank == 0:
        init_device = gang.device
    else:
        init_device = META

    processor = WhisperProcessor.from_pretrained(config.model_name)
    model = WhisperForConditionalGeneration.from_pretrained(config.model_name).to(
        init_device
    )

    ds_builder = load_dataset_builder(config.dataset_name)
    ds = prepare_dataset(
        config,
        lambda x: {
            "audio": processor(
                x["audio"],
                sampling_rate=ds_builder.info.features["audio"].sampling_rate,
                return_tensors="pt",
            ).input_features.squeeze(0)
        },
    )

    pipeline_reader = create_hf_reader(
        dataset=ds,
        gang=gang,
        converter=lambda x: to_batch(x, "whisper", init_device),
        batching=StaticBatching(config.max_num_elements),
        num_prefetch=config.num_prefetch,
        max_seq_len=config.max_audio_len,
    )

    wall_watch = Stopwatch(start=True, device=init_device)

    return HFEvaluator[Seq2SeqBatch](
        model=cast(Model, HGModelWrapper(model)),
        metrics=["bleu"],
        gang=gang,
        data_reader=pipeline_reader,
        wall_watch=wall_watch,
        preprocessor=evaluator_preprocessor,
        postprocessor=lambda x, y: (
            processor.batch_decode(x, skip_special_tokens=True),
            cast(list[str], y.seqs),
        ),
    )
