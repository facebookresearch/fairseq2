# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, TextIO, Union, final, Tuple
from torch import Tensor

from fairseq2.datasets.mmlu import load_mmlu_dataset, MMLUBatch
from fairseq2.recipes.common_metrics import RepresentationAlignMetricBag
from fairseq2.models.sequence import (
    SequenceBatch,
    SequenceModel,
    SequenceModelOutput,
    as_auto_regressive_input,
    SpeechTextReprOutput,
    SpeechTextPPLOutput
)

import torch
from torch.nn import Module
from fairseq2.checkpoint import CheckpointModelMetadataProvider, FileCheckpointManager
from fairseq2.assets import default_asset_store
from fairseq2.assets.utils import retrieve_asset_card
from fairseq2.checkpoint import CheckpointModelMetadataProvider
from fairseq2.config_registry import ConfigRegistry
from fairseq2.data.text import TextTokenDecoder, TextTokenizer, load_text_tokenizer
from fairseq2.datasets import LengthBatching
from fairseq2.datasets.asr import load_asr_dataset
from fairseq2.gang import Gang
from fairseq2.logging import get_log_writer
from fairseq2.metrics.wer import WerMetric
from fairseq2.models.seq2seq import Seq2SeqBatch
from fairseq2.nn.padding import PaddingMask
from fairseq2.models.wav2vec2.asr import Wav2Vec2AsrModel, load_wav2vec2_asr_model
from fairseq2.models.llama.loader import load_llama_model
from fairseq2.models.wav2vec2.asr.model import Wav2Vec2AsrOutput
from fairseq2.nn.utils.module import remove_parametrizations
from fairseq2.recipes.evaluator import AbstractEvalUnit, Evaluator
from fairseq2.recipes.utils.log import log_model
from fairseq2.recipes.utils.setup import (
    broadcast_model,
    check_model_type,
    setup_root_gang,
)
from fairseq2.recipes.speech_llm.metrics import PPLEvalMetricBag, MMLUEvalMetricBag
from fairseq2.recipes.speech_llm.speech_text_align import SpeechTextAlignEvalUnit
from fairseq2.recipes.wav2vec2.asr.common import Wav2Vec2AsrMetricBag
from fairseq2.typing import META, DataType, override
from fairseq2.utils.profiler import Stopwatch

log = get_log_writer(__name__)


@dataclass
class SpeechTextEvalConfig:
    """Holds the configuration of a wav2vec 2.0 ASR model evaluation task."""

    # Data
    dataset: Union[str, Path] = "librispeech_align"
    """The name or path to the asset card of the ASR dataset."""

    max_seq_len: int = 1024
    """The maximum sequence length."""
    min_seq_len: int = 5

    max_num_tokens: int = 1024
    """The maximum number of tokens per batch."""

    num_prefetch: int = 4
    """The number of batches to prefetch in background."""

    # Model
    model: Union[str, Path] = "speech_llama3_8b"
    """The name or path to the asset card of the wav2vec 2.0 ASR model to evaluate."""

    checkpoint_dir: Optional[Path] = None
    """The checkpoint directory containing models saved by :class:`FileCheckpointManager`."""

    dtype: DataType = torch.bfloat16
    """The data type of the model."""
    # Misc
    seed: int = 2
    """The random number generator seed to use."""
    num_shots: Optional[int] = None

mmlu_eval_presets = ConfigRegistry[SpeechTextEvalConfig]()

mmlu_eval_preset = mmlu_eval_presets.decorator


@mmlu_eval_preset("speech_mmlu")
def _base_mmlu() -> SpeechTextEvalConfig:
    config = SpeechTextEvalConfig()
    config.checkpoint_dir = Path("/fsx-checkpoints/steventan0110/tmp/preset_llama3_8b_speech_text_align.ws_8/checkpoints")
    config.dataset = "mmlu"
    config.num_shots = 5
    return config



def load_mmlu_evaluator(
    config: SpeechTextEvalConfig, output_dir: Path
) -> Evaluator[Seq2SeqBatch]:
    """Load an :class:`Evaluator` for wav2vec 2.0 ASR model evaluation."""
    wall_watch = Stopwatch(start=True)


    if config.checkpoint_dir is not None:
        default_asset_store.metadata_providers.append(
            CheckpointModelMetadataProvider(config.checkpoint_dir)
        )

    gang = setup_root_gang(log)

    checkpoint_manager = FileCheckpointManager(
        config.checkpoint_dir, gang
    )

    # Load the tokenizer.
    model_card = retrieve_asset_card(config.model)

    log.info("Loading {} tokenizer.", model_card.name)

    tokenizer = load_text_tokenizer(model_card)

    log.info("Tokenizer loaded.")

    # Load the dataset.
    dataset_card = retrieve_asset_card(config.dataset)

    log.info("Loading {} dataset.", dataset_card.name)

    dataset = load_mmlu_dataset(dataset_card)
    
    log.info("Dataset loaded.")

    text_dataset_reader = dataset.create_reader(
        tokenizer,
        gang,
        num_prefetch=config.num_prefetch,
        seed=config.seed,
        mode="text"
    )

    speech_dataset_reader = dataset.create_reader(
        tokenizer,
        gang,
        num_prefetch=config.num_prefetch,
        seed=config.seed,
        mode="speech"
    )

    speech_text_dataset_reader = dataset.create_reader(
        tokenizer,
        gang,
        num_prefetch=config.num_prefetch,
        seed=config.seed,
        mode="speech_text"
    )
    # for item in speech_text_dataset_reader["management"]:
    #     print(item)
    #     exit(0)

    # for item in speech_dataset_reader["management"]:
    #     print(item)
    #     exit(0)

    # Load the model.
    log.info("Loading {} model on rank 0.", model_card.name)

    if gang.rank == 0:
        init_device = gang.device
    else:
        init_device = META

    has_checkpoint = checkpoint_manager.has_checkpoint()

    if has_checkpoint:
        log.info("Loading from checkpoint dir")
        model = load_llama_model(model_card, device=init_device, dtype=config.dtype)
    else:
        log.info("No checkpoint from specified dir: {}", config.checkpoint_dir)
        exit(0)
  
    gang.barrier()

    log.info("Model loaded on rank 0.")
    remove_parametrizations(model)

    # Distribute the model to all processes in the gang.
    if gang.size != 1:
        broadcast_model(model, gang, log)

    task_to_eval = "management"
    text_unit = TextMMLUEval(model, gang, task_to_eval, display_name=f"text_mmlu_{task_to_eval}")
    speech_unit = SpeechMMLUEval(model, gang, task_to_eval, display_name=f"speech_mmlu_{task_to_eval}")
    # Initialize the evaluator.
    return Evaluator[Seq2SeqBatch](
        units=[speech_unit, speech_unit],
        data_readers=[speech_dataset_reader[task_to_eval], speech_text_dataset_reader[task_to_eval]],
        root_gang=gang,
        tb_dir=output_dir.joinpath("tb"),
        metrics_dir=output_dir.joinpath("metrics"),
        seed=config.seed,
        wall_watch=wall_watch,
    )


@final
class TextMMLUEval(AbstractEvalUnit[MMLUBatch]):
    _metric_bag: MMLUEvalMetricBag

    def __init__(self, model: Module, gang: Gang, task: str, display_name: Optional[str]=None) -> None:
        super().__init__(model, display_name=display_name)
        self._metric_bag = MMLUEvalMetricBag(gang)
        self.task = task

    @override
    def __call__(self, batch: MMLUBatch) -> Tuple[Tensor, int]:
        output = self._forward(batch)
        nll, acc = output.compute_acc(batch.answer)
        self._metric_bag.update_nll(acc, nll)
       

    @torch.inference_mode()
    def _forward(self, batch: MMLUBatch) -> SpeechTextPPLOutput:
        input_seqs = batch.input_tokens
        output_seqs = batch.output_tokens
        target_mask = batch.output_tokens != -100
        return self._model.forward_text_nll(
            input_seqs, 
            output_seqs, 
            target_mask,
            None
        )
    
    @property
    @override
    def metric_bag(self) -> PPLEvalMetricBag:
        return self._metric_bag
    

@final
class SpeechMMLUEval(AbstractEvalUnit[MMLUBatch]):
    _metric_bag: MMLUEvalMetricBag

    def __init__(self, model: Module, gang: Gang, task: str, display_name: Optional[str]=None) -> None:
        """
        :param model:
            The language model. Might be wrapped with DDP or FSDP.
        :param gang:
            The gang for distributed training.
        """
        super().__init__(model, display_name=display_name)
        self._metric_bag = MMLUEvalMetricBag(gang)


    @override
    def __call__(self, batch: MMLUBatch) -> Tuple[Tensor, int]:
        output = self._forward(batch)
        nll, acc = output.compute_acc(batch.answer)
        self._metric_bag.update_nll(acc, nll)
       

    @torch.inference_mode()
    def _forward(self, batch: MMLUBatch) -> SpeechTextPPLOutput:
        # print(batch)
        return self._model.forward_nll_mmlu(
            batch.audios,
            batch.boundary_index, 
            batch.text_seq_lens,
            batch.input_tokens,
            batch.output_tokens,
            batch.speech_positions
        )

    @property
    @override
    def metric_bag(self) -> MMLUEvalMetricBag:
        return self._metric_bag

    



