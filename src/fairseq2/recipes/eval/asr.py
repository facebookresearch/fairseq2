# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass
import math
from pathlib import Path
from typing import Callable, Optional, cast
import torch

from fairseq2.data.data_pipeline import SequenceData
from fairseq2.data.text import load_text_tokenizer
from fairseq2.nn.padding import get_seqs_and_padding_mask
from fairseq2.recipes.utils.setup import setup_root_gang
from fairseq2.datasets.huggingface import Example, create_hf_reader, BatcherBySize
from fairseq2.models.seq2seq import Seq2SeqBatch
from fairseq2.recipes.evaluator import HFEvaluator
from fairseq2.recipes.eval.configs import HFEvalConfig, hf_presets
from fairseq2.models.wav2vec2.asr import load_wav2vec2_asr_model
from fairseq2.typing import META, DataType
from fairseq2.logging import get_log_writer
from datasets import load_dataset, Dataset
from evaluate import load as load_metric

log = get_log_writer(__name__)

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
    source_data = cast(SequenceData, examples['audio'])
    target_data = cast(SequenceData, examples['text'])

    source_seqs, source_padding_mask = get_seqs_and_padding_mask(source_data)
    target_seqs, target_padding_mask = get_seqs_and_padding_mask(target_data)

    return Seq2SeqBatch(
        source_seqs,
        source_padding_mask,
        target_seqs,
        target_padding_mask,
        examples,
    )

@hf_presets.decorator("librispeech_asr")
def _librispeech_asr_config() -> AsrEvalConfig:
    return AsrEvalConfig(
        dataset_name="librispeech_asr",
        model_name="wav2vec2_asr_base_10h",
        split="test.other"
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
    
    # Load dataset
    iterable_ds = load_dataset(config.dataset_name, split=config.split, streaming=True)
    max_samples = config.max_samples if config.max_samples is not None else math.inf
    # Load a subset of the dataset if max_samples is set
    ds = Dataset.from_generator(lambda: (yield from (item for idx, item in enumerate(iterable_ds) if idx < max_samples)), features=iterable_ds.features)

    # Setup GANG
    gang = setup_root_gang(log)

    # Load tokenizer
    tokenizer = load_text_tokenizer(config.tokenizer_name)

    encoder = tokenizer.create_encoder(device=gang.device)
    decoder = tokenizer.create_decoder()

    # Preprocess dataset
    def _preprocess_example(example):
        """
        Preprocesses an individual example by converting the audio array to a PyTorch tensor
        and encoding the text.

        Args:
            example (dict): A dictionary containing "audio" and "text" keys.

        Returns:
            dict: A dictionary with "audio" and "text" as PyTorch tensors.
        """
        audio_tensor = torch.from_numpy(example["audio"]['array']).to(torch.float16).to(gang.device)
        text_tensor = encoder(example['text'].lower()).to(gang.device)
        return {"audio": audio_tensor, "text": text_tensor}
    
    ds = ds.map(_preprocess_example)
    ds.set_format("torch", columns=['audio', 'text'])

    # Create data pipeline from dataset
    batcher = BatcherBySize(bucket_size=config.max_num_elements)
    pipeline = create_hf_reader(dataset=ds, gang=gang, converter=_librispeech_asr_to_batch, batcher=batcher, num_prefetch=config.num_prefetch, pad_value=tokenizer.vocab_info.pad_idx)

    # Load model
    if gang.rank == 0:
        init_device = gang.device
    else:
        init_device = META

    model = load_wav2vec2_asr_model(config.model_name, device=init_device, dtype=config.dtype)

    # Load BLEU from evaluate library
    # bleu = load_metric("bleu")
    
    raise HFEvaluator[Seq2SeqBatch](
        model=model,
        metrics=["bleu"],
        gang=gang,
        data_reader=pipeline,
    )
