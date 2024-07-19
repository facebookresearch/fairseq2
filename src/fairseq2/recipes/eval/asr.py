# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch

from fairseq2.datasets.huggingface import Example
from fairseq2.models.seq2seq import Seq2SeqBatch
from fairseq2.recipes.evaluator import HFEvaluator
from fairseq2.recipes.eval.configs import HFEvalConfig, hf_presets
from fairseq2.typing import DataType


@dataclass
class AsrEvalConfig(HFEvalConfig):
    """Holds the configuration of a ASR evaluation recipe."""

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

    max_samples: Optional[int] = None
    """Maximum number of samples from the dataset to be evaluated. Used
    e.g. for debugging. Default is None, meaning all samples will be evaluated"""

    num_prefetch: int = 4
    """The number of batches to prefetch in background."""

    checkpoint_dir: Optional[Path] = None
    """The checkpoint directory containing models saved by a :class:`FileCheckpointManager`."""

    dtype: DataType = torch.float16
    """The data type of the model."""

def _librispeech_asr_to_batch(examples: Example) -> Seq2SeqBatch:
    # FIXME: Implement the function to convert the collated data loaded from HF dataset
    # "librispeech_asr" to Seq2SeqBatch
    raise NotImplementedError()

@hf_presets.decorator("librispeech_asr")
def _librispeech_asr_config() -> AsrEvalConfig:
    return AsrEvalConfig(
        dataset_name="librispeech_asr",
        model_name="wav2vec2_asr_base_10h",
        # converter=librispeech_asr_to_batch,
    )


def load_wav2vec2_asr_evaluator(
    config: HFEvalConfig, output_dir: Path
) -> HFEvaluator[Seq2SeqBatch]:
    """
    Load the evaluator used for downstream evaluation of the model 
    in a downstream dataset and report BLEU scores
    """
    if not isinstance(config, AsrEvalConfig):
        raise ValueError(f"Expect AsrEvalConfig, get {type(config)}")

    ##########################################################################
    # FIXME: Implement the evaluator loader that:
    # - Construct the DataPipelineReader from its dataset_name and split
    #   using fairseq2.datasets.huggingface.create_hf_reader()
    # - Load wav2vec2 ASR model where the model card is given in `model_name`
    # - Load BLEU from evaluate library
    # - Build the HFEvaluator accordingly
    ##########################################################################
    raise NotImplementedError()
