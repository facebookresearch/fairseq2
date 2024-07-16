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

from fairseq2.datasets.speech_text import load_speech_text_dataset, SpeechTextAlignBatch
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
from fairseq2.recipes.speech_llm.metrics import PPLEvalMetricBag
from fairseq2.recipes.speech_llm.speech_text_align import SpeechTextAlignEvalUnit
from fairseq2.recipes.wav2vec2.asr.common import Wav2Vec2AsrMetricBag
from fairseq2.typing import META, DataType, override
from fairseq2.utils.profiler import Stopwatch
from fairseq2.datasets.speech_text import load_align_speech_text_librispeech_dataset
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


speech_text_eval_presets = ConfigRegistry[SpeechTextEvalConfig]()

speech_text_eval_preset = speech_text_eval_presets.decorator


@speech_text_eval_preset("librispeech_similarity")
def _base_similarity() -> SpeechTextEvalConfig:
    config = SpeechTextEvalConfig()
    config.checkpoint_dir = Path("/fsx-checkpoints/steventan0110/tmp/preset_llama3_8b_speech_text_align.ws_8/checkpoints")
    return config


def load_speech_text_evaluator(
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

    dataset = load_align_speech_text_librispeech_dataset(dataset_card)

    log.info("Dataset loaded.")

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
  
    # model = load_llama_model(model_card, device=init_device, dtype=config.dtype)

    gang.barrier()

    log.info("Model loaded on rank 0.")
    remove_parametrizations(model)

    # Distribute the model to all processes in the gang.
    if gang.size != 1:
        broadcast_model(model, gang, log)

    wer_unit = SpeechTextAlignEvalASRUnit(model, gang, tokenizer, "librispeech_align")
    ppl_unit = SpeechTextPPLEval(model, gang, tokenizer.vocab_info.bos_idx, "librispeech_ppl")
    text_ppl_unit = TextPPLEval(model, gang, tokenizer.vocab_info.bos_idx, "text_ppl")
    
    wer_reader = dataset.create_reader(
        tokenizer,
        gang,
        config.max_seq_len,
        config.max_num_tokens,
        config.min_seq_len,
        num_prefetch=config.num_prefetch,
        seed=config.seed,
    )
    ppl_reader = dataset.create_reader(
        tokenizer,
        gang,
        config.max_seq_len,
        config.max_num_tokens,
        config.min_seq_len,
        num_prefetch=config.num_prefetch,
        seed=config.seed,
    )
    text_ppl_reader = dataset.create_reader(
        tokenizer,
        gang,
        config.max_seq_len,
        config.max_num_tokens,
        config.min_seq_len,
        num_prefetch=config.num_prefetch,
        seed=config.seed,
    )


    # Initialize the evaluator.
    return Evaluator[Seq2SeqBatch](
        units=[text_ppl_unit, wer_unit, ppl_unit],
        data_readers=[text_ppl_reader, wer_reader, ppl_reader],
        # units=[text_ppl_unit],
        # data_readers=[text_ppl_reader],
        root_gang=gang,
        tb_dir=output_dir.joinpath("tb"),
        metrics_dir=output_dir.joinpath("metrics"),
        seed=config.seed,
        wall_watch=wall_watch,
    )


@final
class TextPPLEval(AbstractEvalUnit[SpeechTextAlignBatch]):
    _metric_bag: PPLEvalMetricBag

    def __init__(self, model: Module, gang: Gang, bos_id: int, display_name: Optional[str]=None) -> None:
        super().__init__(model, display_name=display_name)
        self._metric_bag = PPLEvalMetricBag(gang)
        self.bos_id = bos_id

    @override
    def __call__(self, batch: SpeechTextAlignBatch) -> Tuple[Tensor, int]:
        output = self._forward(batch)
        nll, num_elements = output.compute_loss()
        self._metric_bag.update_nll(nll, num_elements)
        self._metric_bag.update_batch_metrics(batch.text_tokens)
       

    @torch.inference_mode()
    def _forward(self, batch: SpeechTextAlignBatch) -> SpeechTextPPLOutput:
        input_seqs, output_seqs, target_mask, padding_mask = self.get_input_output_texts(batch, self.bos_id)
        return self._model.forward_text_nll(
            input_seqs, 
            output_seqs, 
            target_mask,
            padding_mask
        )
    
    @staticmethod
    def get_input_output_texts(batch: SpeechTextAlignBatch, bos_id: int) -> None:
        text_seqs, padding_mask = batch.text_tokens.seqs, batch.text_tokens.padding_mask
        prepend_bos_token = text_seqs.new_full((text_seqs.shape[0], 1), bos_id)
        text_seqs = torch.cat((prepend_bos_token, text_seqs), dim=1)

        input_seqs, output_seqs = text_seqs[:, :-1], text_seqs[:, 1:]
        if padding_mask:
            target_mask = padding_mask.materialize()
           
            new_padding_seq_lens, new_padding_batch_size = padding_mask.seq_lens + 1, padding_mask._batch_seq_len
            new_padding_seq_lens[new_padding_seq_lens > new_padding_batch_size] = new_padding_batch_size
            new_padding_mask = PaddingMask(new_padding_seq_lens, new_padding_batch_size)
        else:
            # no padding for the current batch
            target_mask = input_seqs.new_full((input_seqs.shape[0], input_seqs.shape[1]), True).bool()
            new_padding_mask = None # model can deal with this and automatically use full mask
        return input_seqs, output_seqs, target_mask, new_padding_mask

    @property
    @override
    def metric_bag(self) -> PPLEvalMetricBag:
        return self._metric_bag
    

@final
class SpeechTextPPLEval(AbstractEvalUnit[SpeechTextAlignBatch]):
    _metric_bag: PPLEvalMetricBag

    def __init__(self, model: Module, gang: Gang, bos_id: int, display_name: Optional[str]=None) -> None:
        """
        :param model:
            The language model. Might be wrapped with DDP or FSDP.
        :param gang:
            The gang for distributed training.
        """
        super().__init__(model, display_name=display_name)
        self._metric_bag = PPLEvalMetricBag(gang)
        self.bos_id = bos_id

    @override
    def __call__(self, batch: SpeechTextAlignBatch) -> Tuple[Tensor, int]:
        output = self._forward(batch)
        nll, num_elements = output.compute_loss()
        self._metric_bag.update_nll(nll, num_elements)
        self._metric_bag.update_batch_metrics(batch.text_tokens)
       

    @torch.inference_mode()
    def _forward(self, batch: SpeechTextAlignBatch) -> SpeechTextPPLOutput:
        input_seqs, output_seqs, target_mask, padding_mask = self.get_input_output_texts(batch, self.bos_id)
        return self._model.forward_nll(
            batch.audios,
            batch.text_tokens,
            batch.boundary_index, 
            input_seqs, 
            output_seqs, 
            target_mask,
            padding_mask
        )
    
    @staticmethod
    def get_input_output_texts(batch: SpeechTextAlignBatch, bos_id: int) -> None:
        text_seqs, padding_mask = batch.text_tokens.seqs, batch.text_tokens.padding_mask
        prepend_bos_token = text_seqs.new_full((text_seqs.shape[0], 1), bos_id)
        text_seqs = torch.cat((prepend_bos_token, text_seqs), dim=1)

        input_seqs, output_seqs = text_seqs[:, :-1], text_seqs[:, 1:]
        if padding_mask:
            target_mask = padding_mask.materialize()
           
            new_padding_seq_lens, new_padding_batch_size = padding_mask.seq_lens + 1, padding_mask._batch_seq_len
            new_padding_seq_lens[new_padding_seq_lens > new_padding_batch_size] = new_padding_batch_size
            new_padding_mask = PaddingMask(new_padding_seq_lens, new_padding_batch_size)

        else:
            # no padding for the current batch
            target_mask = input_seqs.new_full((input_seqs.shape[0], input_seqs.shape[1]), True).bool()
            new_padding_mask = None # model can deal with this and automatically use full mask
        
        return input_seqs, output_seqs, target_mask, new_padding_mask


    @property
    @override
    def metric_bag(self) -> PPLEvalMetricBag:
        return self._metric_bag

    

@final
class SpeechTextAlignEvalASRUnit(AbstractEvalUnit[SpeechTextAlignBatch]):
    """Represents a language model instruction-finetuning unit."""

    _metric_bag: RepresentationAlignMetricBag

    def __init__(self, model: Module, gang: Gang, tokenizer: TextTokenizer, display_name: Optional[str]=None) -> None:
        """
        :param model:
            The language model. Might be wrapped with DDP or FSDP.
        :param gang:
            The gang for distributed training.
        """
        super().__init__(model, display_name=display_name)
        self._metric_bag = RepresentationAlignMetricBag(gang)
        self.tokenizer = tokenizer.create_decoder()

    @override
    def __call__(self, batch: SpeechTextAlignBatch) -> Tuple[Tensor, int]:
        output = self._forward(batch)
        embed_table = self._model.decoder_frontend.embed.weight
        text_tokens = batch.text_tokens.seqs
        out = output.compute_asr(embed_table=embed_table, text_tokens=text_tokens, tokenizer=self.tokenizer)
        wer, wer_lens = out["wer"], out["wer_lens"]
        wer_score = wer / wer_lens
        self._metric_bag.update_wer(wer_score, wer_lens)
        self._metric_bag.update_batch_metrics(batch.text_tokens)
       
    def _forward(self, batch: SpeechTextAlignBatch) -> SpeechTextReprOutput:
        return self._model(batch.audios, batch.text_tokens, batch.boundary_index)

    @property
    @override
    def metric_bag(self) -> RepresentationAlignMetricBag:
        return self._metric_bag

