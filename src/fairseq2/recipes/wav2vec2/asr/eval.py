# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, TextIO, Union, final

import torch
from torch.nn import Module

from fairseq2.assets import default_asset_store
from fairseq2.assets.utils import retrieve_asset_card
from fairseq2.checkpoint import CheckpointModelMetadataProvider
from fairseq2.config_registry import ConfigRegistry
from fairseq2.data.text import TextTokenDecoder, TextTokenizer, load_text_tokenizer
from fairseq2.datasets.asr import load_asr_dataset
from fairseq2.gang import Gang
from fairseq2.logging import get_log_writer
from fairseq2.metrics.wer import WerMetric
from fairseq2.models.seq2seq import Seq2SeqBatch
from fairseq2.models.sequence import SequenceBatch
from fairseq2.models.wav2vec2.asr import Wav2Vec2AsrModel, load_wav2vec2_asr_model
from fairseq2.models.wav2vec2.asr.model import Wav2Vec2AsrOutput
from fairseq2.nn.utils.module import remove_parametrizations
from fairseq2.recipes.evaluator import AbstractEvalUnit, Evaluator
from fairseq2.recipes.utils.log import log_model
from fairseq2.recipes.utils.setup import (
    broadcast_model,
    check_model_type,
    setup_root_gang,
)
from fairseq2.recipes.wav2vec2.asr.common import Wav2Vec2AsrMetricBag
from fairseq2.typing import META, DataType, override
from fairseq2.utils.profiler import Stopwatch

log = get_log_writer(__name__)


@dataclass
class Wav2Vec2AsrEvalConfig:
    """Holds the configuration of a wav2vec 2.0 ASR model evaluation task."""

    # Data
    dataset: Union[str, Path] = "librilight_asr_10h"
    """The name or path to the asset card of the ASR dataset."""

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


@wav2vec2_asr_eval_preset("base_10h")
def _base_10h() -> Wav2Vec2AsrEvalConfig:
    return Wav2Vec2AsrEvalConfig()


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
    output_file = output_dir.joinpath(f"transcriptions/rank_{gang.rank}.txt")

    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
    except OSError as ex:
        raise RuntimeError(
            f"The output directory ({output_file.parent}) cannot be created. See nested exception for details."
        ) from ex

    try:
        output_fp = output_file.open("w")
    except OSError as ex:
        raise RuntimeError(
            f"The output file ({output_file}) cannot be created. See nested exception for details."
        ) from ex

    unit = Wav2Vec2AsrEvalUnit(model, gang, tokenizer, output_stream=output_fp)

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
        tb_dir=output_dir.joinpath("tb"),
        metrics_dir=output_dir.joinpath("metrics"),
        seed=config.seed,
        wall_watch=wall_watch,
    )


@final
class Wav2Vec2AsrEvalUnit(AbstractEvalUnit[Seq2SeqBatch]):
    """Represents a wav2vec 2.0 ASR model evaluation unit."""

    _metric_bag: Wav2Vec2AsrEvalMetricBag

    def __init__(
        self,
        model: Module,
        gang: Gang,
        tokenizer: TextTokenizer,
        *,
        output_stream: Optional[TextIO] = None,
    ) -> None:
        """
        :param model:
            The wav2vec 2.0 ASR model. Might be wrapped with DDP or FSDP.
        :param gang:
            The gang for distributed evaluation.
        :param tokenizer:
            The tokenizer to use.
        :param output_stream:
            The output stream to dump transcriptions, WER, and UER metrics.
        """
        super().__init__(model)

        check_model_type(model, Wav2Vec2AsrModel)

        self._metric_bag = Wav2Vec2AsrEvalMetricBag(
            gang, tokenizer, output_stream=output_stream
        )

    @override
    def __call__(self, batch: Seq2SeqBatch) -> None:
        input_batch = SequenceBatch(batch.source_seqs, batch.source_padding_mask)

        output = self._forward(input_batch)

        loss = output.compute_loss(batch.target_seqs, batch.target_padding_mask)

        self._metric_bag.update_ctc_loss(batch, loss.detach())

        self._metric_bag.update_batch_metrics(batch)

        self._metric_bag.update_wer(batch, output)

    def _forward(self, batch: SequenceBatch) -> Wav2Vec2AsrOutput:
        return self._model(batch)  # type: ignore[no-any-return]

    @property
    @override
    def metric_bag(self) -> Wav2Vec2AsrEvalMetricBag:
        return self._metric_bag

    @property
    @override
    def throughput_metric_name(self) -> Optional[str]:
        return "num_source_elements"


class Wav2Vec2AsrEvalMetricBag(Wav2Vec2AsrMetricBag):
    """Holds the metrics of a wav2vec 2.0 ASR model evaluation task."""

    _wer: WerMetric
    _text_decoder: TextTokenDecoder
    _pad_idx: int
    _blank_label: int
    _output_stream: Optional[TextIO]

    def __init__(
        self,
        gang: Gang,
        tokenizer: TextTokenizer,
        *,
        blank_label: int = 0,
        output_stream: Optional[TextIO] = None,
    ) -> None:
        """
        :param gang:
            The gang over which to sync metrics.
        :param tokenizer:
            The text tokenizer to compute the WER score.
        :param blank_label:
            The blank label in logits.
        :param output_stream:
            The output stream to dump transcriptions, WER, and UER metrics.
        """
        super().__init__(gang)

        self.register_metric("_wer", WerMetric(device=gang.device), persistent=False)

        self._text_decoder = tokenizer.create_decoder()

        pad_idx = tokenizer.vocab_info.pad_idx
        if pad_idx is None:
            raise ValueError(
                "``vocab_info` of `tokenizer` must have a PAD symbol defined."
            )

        self._pad_idx = pad_idx

        self._blank_label = blank_label

        self._output_stream = output_stream

    @torch.inference_mode()
    def update_wer(self, batch: Seq2SeqBatch, model_output: Wav2Vec2AsrOutput) -> None:
        """Update the WER (Word Error Rate) score metric.

        :param batch:
            The batch processed by the model.
        :param model_output:
            The output of the model for ``batch``.
        """
        # (N, S), (N, S)
        hyp_seqs, hyp_padding_mask = model_output.generate_hypotheses(
            self._pad_idx, self._blank_label
        )

        self._wer.update(
            self._text_decoder,
            batch.target_seqs,
            batch.target_padding_mask,
            hyp_seqs,
            hyp_padding_mask,
            output_stream=self._output_stream,
        )

        if self._output_stream is not None:
            self._output_stream.flush()

    @override
    def process_metric_values(self, values: Dict[str, Any]) -> None:
        super().process_metric_values(values)

        uer, wer = values.pop("wer")

        values["uer"] = uer
        values["wer"] = wer
