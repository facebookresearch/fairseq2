# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO, final

import torch
from torch.nn import Module
from typing_extensions import override

from fairseq2.assets import AssetNotFoundError, default_asset_store
from fairseq2.checkpoint import CheckpointModelMetadataProvider
from fairseq2.config_registry import ConfigRegistry
from fairseq2.data.text import TextTokenDecoder, TextTokenizer, load_char_tokenizer
from fairseq2.datasets import LengthBatching
from fairseq2.datasets.asr import GenericAsrDataset, load_asr_dataset
from fairseq2.gang import Gang
from fairseq2.logging import get_log_writer
from fairseq2.metrics.text import WerMetric
from fairseq2.models import load_model
from fairseq2.models.seq2seq import Seq2SeqBatch
from fairseq2.models.sequence import SequenceBatch
from fairseq2.models.wav2vec2.asr import Wav2Vec2AsrModel
from fairseq2.models.wav2vec2.asr.model import Wav2Vec2AsrOutput
from fairseq2.nn.utils.module import remove_parametrizations
from fairseq2.recipes.evaluator import AbstractEvalUnit, Evaluator
from fairseq2.recipes.utils.asset import (
    AssetReference,
    asset_as_path,
    retrieve_asset_card,
)
from fairseq2.recipes.utils.log import log_model
from fairseq2.recipes.utils.setup import (
    broadcast_model,
    check_model_type,
    setup_root_gang,
)
from fairseq2.recipes.wav2vec2.asr.common import Wav2Vec2AsrMetricBag
from fairseq2.typing import META, DataType
from fairseq2.utils.profiler import Stopwatch

log = get_log_writer(__name__)


@dataclass(kw_only=True)
class Wav2Vec2AsrEvalConfig:
    """Holds the configuration of a wav2vec 2.0 ASR model evaluation task."""

    # Data
    dataset: AssetReference = "librilight_asr_10h"
    """The name, path, or path to the asset card of the ASR dataset."""

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
    model: AssetReference = "wav2vec2_asr_base_10h"
    """The name or path to the asset card of the wav2vec 2.0 ASR model to evaluate."""

    checkpoint_dir: Path | None = None
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
            CheckpointModelMetadataProvider(
                config.checkpoint_dir, lower_score_better=True
            )
        )

    gang = setup_root_gang(log)

    model_card = retrieve_asset_card(config.model)

    # Load the tokenizer.
    log.info("Loading {} tokenizer.", model_card.name)

    tokenizer = load_char_tokenizer(model_card)

    log.info("Tokenizer loaded.")

    # Load the dataset.
    try:
        dataset_card = retrieve_asset_card(config.dataset)
    except AssetNotFoundError:
        dataset_card = None

    if dataset_card is not None:
        log.info("Loading {} ASR dataset.", dataset_card.name)

        dataset = load_asr_dataset(dataset_card)

        log.info("Dataset loaded.")
    else:
        dataset_path = asset_as_path(config.dataset)

        dataset = GenericAsrDataset.from_path(dataset_path)

    # Load the model.
    log.info("Loading {} model on rank 0.", model_card.name)

    if gang.rank == 0:
        init_device = gang.device
    else:
        init_device = META

    try:
        model = load_model(model_card, device=init_device, dtype=config.dtype)
    except ValueError as ex:
        raise ValueError(
            "The model cannot be initialized. See nested exception for details."
        ) from ex

    if not isinstance(model, Wav2Vec2AsrModel):
        raise ValueError(
            f"The model must be of type `{Wav2Vec2AsrModel}`, but is of type `{type(model)}` instead."
        )

    gang.barrier()

    log.info("Model loaded on rank 0.")

    remove_parametrizations(model)

    # Distribute the model to all processes in the gang.
    if gang.size != 1:
        broadcast_model(model, gang, log)

    log_model(model, log)

    # Initialize the evaluation unit.
    ref_output_file = output_dir.joinpath(f"transcriptions/rank_{gang.rank}.ref.txt")
    hyp_output_file = output_dir.joinpath(f"transcriptions/rank_{gang.rank}.hyp.txt")

    try:
        ref_output_file.parent.mkdir(parents=True, exist_ok=True)
    except OSError as ex:
        raise RuntimeError(
            f"The output directory '{ref_output_file.parent}' cannot be created. See nested exception for details."
        ) from ex

    try:
        ref_output_fp = ref_output_file.open("w")
    except OSError as ex:
        raise RuntimeError(
            f"The output file '{ref_output_file}' cannot be created. See nested exception for details."
        ) from ex

    try:
        hyp_output_fp = hyp_output_file.open("w")
    except OSError as ex:
        raise RuntimeError(
            f"The output file '{hyp_output_file}' cannot be created. See nested exception for details."
        ) from ex

    unit = Wav2Vec2AsrEvalUnit(
        model,
        gang,
        tokenizer,
        ref_output_stream=ref_output_fp,
        hyp_output_stream=hyp_output_fp,
    )

    seed = config.seed

    try:
        data_reader = dataset.create_reader(
            config.split,
            tokenizer,
            gang,
            batching=LengthBatching(config.max_num_elements),
            dtype=config.dtype,
            min_audio_len=config.min_audio_len,
            max_audio_len=config.max_audio_len,
            normalize_audio=config.normalize_audio,
            sync_mode="until_last",
            num_prefetch=config.num_prefetch,
            seed=seed,
        )
    except ValueError as ex:
        raise ValueError(
            "The data reader cannot be initialized. See nested exception for details."
        ) from ex

    seed += 1

    # Initialize the evaluator.
    return Evaluator[Seq2SeqBatch](
        units=[unit],
        data_readers=[data_reader],
        root_gang=gang,
        tb_dir=output_dir.joinpath("tb"),
        metrics_dir=output_dir.joinpath("metrics"),
        seed=seed,
        wall_watch=wall_watch,
    )


@final
class Wav2Vec2AsrEvalUnit(AbstractEvalUnit[Seq2SeqBatch]):
    """Represents a wav2vec 2.0 ASR model evaluation unit."""

    _text_decoder: TextTokenDecoder
    _pad_idx: int
    _blank_label: int
    _ref_output_stream: TextIO | None
    _hyp_output_stream: TextIO | None
    _metric_bag: Wav2Vec2AsrEvalMetricBag

    def __init__(
        self,
        model: Module,
        gang: Gang,
        tokenizer: TextTokenizer,
        *,
        blank_label: int = 0,
        ref_output_stream: TextIO | None = None,
        hyp_output_stream: TextIO | None = None,
    ) -> None:
        """
        :param model:
            The wav2vec 2.0 ASR model. Might be wrapped with DDP or FSDP.
        :param gang:
            The gang for distributed evaluation.
        :param tokenizer:
            The tokenizer to encode target text.
        :param blank_label:
            The blank label in logits.
        :param ref_output_stream:
            The output stream to dump references.
        :param hyp_output_stream:
            The output stream to dump hypotheses.
        """
        super().__init__(model)

        check_model_type(model, Wav2Vec2AsrModel)

        self._text_decoder = tokenizer.create_decoder()

        pad_idx = tokenizer.vocab_info.pad_idx
        if pad_idx is None:
            raise ValueError(
                "``vocab_info` of `tokenizer` must have a PAD symbol defined."
            )

        self._pad_idx = pad_idx

        self._blank_label = blank_label

        self._ref_output_stream = ref_output_stream
        self._hyp_output_stream = hyp_output_stream

        self._metric_bag = Wav2Vec2AsrEvalMetricBag(gang)

    @override
    def __call__(self, batch: Seq2SeqBatch) -> None:
        input_batch = SequenceBatch(batch.source_seqs, batch.source_padding_mask)

        output = self._forward(input_batch)

        loss = output.compute_loss(batch.target_seqs, batch.target_padding_mask)

        self._metric_bag.update_ctc_loss(batch, loss)

        self._metric_bag.update_batch_metrics(batch)

        self._compute_wer(batch, output)

    def _compute_wer(self, batch: Seq2SeqBatch, output: Wav2Vec2AsrOutput) -> None:
        # (N, S), (N, S)
        ref_seqs, ref_padding_mask = batch.target_seqs, batch.target_padding_mask

        # (N, S), (N, S)
        hyp_seqs, hyp_padding_mask = output.generate_hypotheses(
            self._pad_idx, self._blank_label
        )

        refs = [self._text_decoder(s) for s in ref_seqs]
        hyps = [self._text_decoder(s) for s in hyp_seqs]

        self._metric_bag.wer.update(
            refs, ref_seqs, ref_padding_mask, hyps, hyp_seqs, hyp_padding_mask
        )

        # Dump references.
        if stream := self._ref_output_stream:
            for ref in refs:
                stream.write(ref)
                stream.write("\n")

            stream.flush()

        # Dump hypotheses.
        if stream := self._hyp_output_stream:
            for hyp in hyps:
                stream.write(hyp)
                stream.write("\n")

            stream.flush()

    def _forward(self, batch: SequenceBatch) -> Wav2Vec2AsrOutput:
        return self._model(batch)  # type: ignore[no-any-return]

    @property
    @override
    def metric_bag(self) -> Wav2Vec2AsrEvalMetricBag:
        return self._metric_bag


class Wav2Vec2AsrEvalMetricBag(Wav2Vec2AsrMetricBag):
    """Holds the metrics of a wav2vec 2.0 ASR model evaluation task."""

    wer: WerMetric

    def __init__(self, gang: Gang) -> None:
        """
        :param gang:
            The gang over which to sync metrics.
        """
        super().__init__(gang, train=False)

        self.register_metric("wer", WerMetric(device=gang.device), persistent=False)

    @override
    def process_metric_values(self, values: dict[str, Any]) -> None:
        super().process_metric_values(values)

        uer, wer = values.pop("wer")

        values["uer"] = uer
        values["wer"] = wer
