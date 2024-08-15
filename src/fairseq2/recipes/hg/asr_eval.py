# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Union, cast

import torch
from datasets import (  # type: ignore[attr-defined,import-untyped,import-not-found]
    Dataset,
    load_dataset,
)

from fairseq2.config_registry import ConfigRegistry
from fairseq2.data.data_pipeline import SequenceData
from fairseq2.data.text import load_text_tokenizer
from fairseq2.data.text.text_tokenizer import TextTokenizer
from fairseq2.datasets.batching import StaticBatching
from fairseq2.logging import get_log_writer
from fairseq2.models.sequence import SequenceBatch
from fairseq2.models.wav2vec2.asr import load_wav2vec2_asr_model
from fairseq2.nn.padding import PaddingMask, get_seqs_and_padding_mask
from fairseq2.recipes.hg.dataset import Example, create_hf_reader
from fairseq2.recipes.hg.evaluator import HFEvaluator
from fairseq2.recipes.utils.setup import setup_root_gang
from fairseq2.typing import META, DataType, Device
from fairseq2.utils.profiler import Stopwatch

log = get_log_writer(__name__)

@dataclass
class AsrDatasetConfig:
    """Configuration for an automatic speech recognition dataset."""

    dataset_path: str
    """The name of the dataset."""

    dataset_name: Optional[str] = None

    source_column: List[str] = field(default_factory=list)
    """The path of the column containing the source audio."""

    target_column: List[str] = field(default_factory=list)
    """The path of the column containing the target text."""

    tokenizer_name: Optional[str] = None
    """The name of the tokenizer to use."""

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'AsrDatasetConfig':
        """Create an AsrDatasetConfig instance from a configuration dictionary."""
        return cls(
            dataset_path=config_dict.get('dataset_path', ''),
            dataset_name=config_dict.get('dataset_name'),
            source_column=config_dict.get('source_column', []),
            target_column=config_dict.get('target_column', []),
            tokenizer_name=config_dict.get('tokenizer_name')
        )

    def get_source_data(self, ds: dict) -> Union[list, dict]:
        """Retrieve the source (audio) data from the dataset."""
        return self._get_data(ds, self.source_column)

    def get_target_data(self, ds: dict) -> Union[list, dict]:
        """Retrieve the target (text) data from the dataset."""
        return self._get_data(ds, self.target_column)

    @staticmethod
    def _get_data(ds: dict, path: List[str]) -> Union[list, dict]:
        """Retrieve data from the dataset using the specified path."""
        current = ds
        for key in path:
            if key in current:
                current = current[key]
            else:
                raise ValueError(f"Invalid path: {path}")
        return current

@dataclass(kw_only=True)
class AsrEvalConfig:
    """Holds the configuration of a ASR evaluation recipe."""

    dataset_config: AsrDatasetConfig
    """The HF dataset to evaluate with."""

    model_name: str
    """The name of the model to evaluate."""

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


@dataclass
class EvalSeqBatch:
    source_seqs: torch.Tensor
    """The source sequences."""

    source_padding_mask: Optional[PaddingMask]
    """The source padding mask."""

    target_seqs: List[str]
    """The target sequences."""


asr_eval_presets = ConfigRegistry[AsrEvalConfig]()
asr_eval_preset = asr_eval_presets.decorator


@asr_eval_preset("default_asr")
def _default_asr_config() -> AsrEvalConfig:
    return AsrEvalConfig(
        dataset_config=AsrDatasetConfig.from_dict({
            'dataset_path': 'librispeech_asr',
            'source_column': ['audio', 'array'],
            'target_column': ['text'],
            'tokenizer_name': 'librispeech_asr',
        }),
        model_name="wav2vec2_asr_base_10h",
        split="test.other",
    )


def extract_features(example: Example, dataset_config: AsrDatasetConfig) -> Example:
    """
    Preprocesses an individual example by converting the audio array to a PyTorch tensor
    and encoding the text.

    Args:
        example (dict): A dictionary containing "audio" and "text" keys.
        device (torch.device): The device to store the tensors.

    Returns:
        dict: A dictionary with "audio" and "text" as PyTorch tensors.
    """
    return {"audio": dataset_config.get_source_data(example), "text": dataset_config.get_target_data(example).lower()}


def to_batch(examples: Example, model_type: str, device: Device) -> EvalSeqBatch:
    """
    Converts a collated batch of examples into a EvalSeqBatch.

    Args:
        examples (dict): A dictionary containing "audio" and "text" keys.

    Returns:
        EvalSeqBatch: A batch of audio and text sequences.
    """

    if model_type == "wav2vec2":
        source_seqs, source_padding_mask = get_seqs_and_padding_mask(examples["audio"])
        source_seqs = source_seqs.to(device)
        source_padding_mask = (
            source_padding_mask.to(device) if source_padding_mask is not None else None
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    target_seqs = examples['text']

    return EvalSeqBatch(
        source_seqs,
        source_padding_mask,
        target_seqs,
    )


def prepare_dataset(
    config: AsrEvalConfig, processor: Optional[Callable[[Example], Example]] = None
) -> Dataset:
    iterable_ds = load_dataset(path=config.dataset_config.dataset_path, name=config.dataset_config.dataset_name, split=config.split, streaming=True)
    ds = Dataset.from_generator(
        lambda: itertools.islice(iterable_ds, 0, config.max_samples),
        features=iterable_ds.features,
    )
    ds = ds.map(lambda x: extract_features(x, config.dataset_config))

    if processor is not None:
        ds = ds.map(processor)

    format = {
        "type": "torch",
        "format_kwargs": {"dtype": config.dtype},
    }
    ds.set_format(**format, columns=["audio", "text"])

    return ds


def evaluator_preprocessor(batch: EvalSeqBatch) -> tuple[SequenceBatch, List[str]]:
    return SequenceBatch(batch.source_seqs, batch.source_padding_mask), batch.target_seqs


def evaluator_postprocesser(
    outputs: Any, targets: list[str], tokenizer: TextTokenizer
) -> tuple[list[str], list[str]]:
    decoder = tokenizer.create_decoder()
    pad_idx = tokenizer.vocab_info.pad_idx

    hypotheses, _ = outputs.generate_hypotheses(pad_idx=pad_idx)
    predictions = [decoder(item) for item in hypotheses]
    references = targets

    return predictions, references


def load_asr_evaluator(
    config: AsrEvalConfig, output_dir: Path
) -> HFEvaluator[EvalSeqBatch]:
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

    ds = prepare_dataset(config)

    gang = setup_root_gang(log)

    if gang.rank == 0:
        init_device = gang.device
    else:
        init_device = META

    tokenizer = load_text_tokenizer(config.dataset_config.tokenizer_name)

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

    return HFEvaluator[EvalSeqBatch](
        model=model,
        metrics=["bleu"],
        gang=gang,
        data_reader=pipeline_reader,
        wall_watch=wall_watch,
        preprocessor=evaluator_preprocessor,
        postprocessor=lambda x, y: evaluator_postprocesser(x, y, tokenizer),
    )
