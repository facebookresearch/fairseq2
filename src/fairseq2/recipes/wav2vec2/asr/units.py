# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, TextIO, Tuple, final

from torch import Tensor
from torch.nn import Module

from fairseq2.data.text import TextTokenizer
from fairseq2.gang import Gang
from fairseq2.logging import get_log_writer
from fairseq2.models.seq2seq import Seq2SeqBatch
from fairseq2.models.sequence import SequenceBatch
from fairseq2.models.wav2vec2.asr import Wav2Vec2AsrModel, Wav2Vec2AsrOutput
from fairseq2.nn.utils.module import freeze_parameters
from fairseq2.recipes.evaluator import AbstractEvalUnit
from fairseq2.recipes.trainer import AbstractTrainUnit
from fairseq2.recipes.utils.setup import check_model_type
from fairseq2.recipes.wav2vec2.asr.metrics import (
    Wav2Vec2AsrEvalMetricBag,
    Wav2Vec2AsrMetricBag,
)
from fairseq2.typing import override

log = get_log_writer(__name__)


@final
class Wav2Vec2AsrTrainUnit(AbstractTrainUnit[Seq2SeqBatch]):
    """Represents the training unit of a wav2vec 2.0 ASR model."""

    _freeze_encoder_for_n_steps: int
    _metric_bag: Wav2Vec2AsrMetricBag

    def __init__(
        self,
        model: Module,
        gang: Gang,
        *,
        freeze_encoder_for_n_steps: int = 0,
    ) -> None:
        """
        :param model:
            The wav2vec 2.0 ASR model. Might be wrapped with DDP or FSDP.
        :param gang:
            The gang for distributed training.
        :param freeze_encoder_for_n_steps:
            The encoder will be frozen for this number of steps.
        """
        super().__init__(model)

        check_model_type(model, Wav2Vec2AsrModel)

        self._freeze_encoder_for_n_steps = freeze_encoder_for_n_steps

        self._metric_bag = Wav2Vec2AsrMetricBag(gang)

    @override
    def __call__(self, batch: Seq2SeqBatch) -> Tuple[Tensor, int]:
        input_batch = SequenceBatch(batch.source_seqs, batch.source_padding_mask)

        output = self._forward(input_batch)

        loss = output.compute_loss(batch.target_seqs, batch.target_padding_mask)

        self._metric_bag.update_loss_metrics(batch, loss.detach())

        return loss, batch.batch_size

    def _forward(self, batch: SequenceBatch) -> Wav2Vec2AsrOutput:
        return self._model(batch)  # type: ignore[no-any-return]

    @override
    def set_step_nr(self, step_nr: int) -> None:
        if isinstance(self._model, Wav2Vec2AsrModel):
            model = self._model
        else:
            model = self._model.module  # DDP or FSDP

        if step_nr <= self._freeze_encoder_for_n_steps:
            if step_nr == 1:
                log.info("Freezing the encoder for the first {} steps.", self._freeze_encoder_for_n_steps)  # fmt: skip

            freeze_parameters(model.encoder_frontend)
            freeze_parameters(model.encoder)

            if model.masker is not None:
                freeze_parameters(model.masker)
        else:
            if step_nr == self._freeze_encoder_for_n_steps + 1:
                log.info("Unfreezing the encoder after step {}.", step_nr - 1)

            freeze_parameters(model, False)

            # We never train the feature extractor.
            freeze_parameters(model.encoder_frontend.feature_extractor)

    @property
    @override
    def metric_bag(self) -> Wav2Vec2AsrMetricBag:
        return self._metric_bag

    @property
    @override
    def throughput_metric_name(self) -> str:
        return "num_source_elements"


@final
class Wav2Vec2AsrEvalUnit(AbstractEvalUnit[Seq2SeqBatch]):
    """Represents the evaluation unit of a wav2vec 2.0 ASR model."""

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

        self._metric_bag.update_loss_metrics(batch, loss.detach())

        self._metric_bag.update_wer(batch, output)

    def _forward(self, batch: SequenceBatch) -> Wav2Vec2AsrOutput:
        return self._model(batch)  # type: ignore[no-any-return]

    @property
    @override
    def metric_bag(self) -> Wav2Vec2AsrEvalMetricBag:
        return self._metric_bag

    @property
    @override
    def throughput_metric_name(self) -> str:
        return "num_source_elements"
