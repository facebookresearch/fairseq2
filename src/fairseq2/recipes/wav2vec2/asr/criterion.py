# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import Optional, Tuple, cast, final

from torch import Tensor
from torch.nn import Module

from fairseq2.data.text import TextTokenizer
from fairseq2.gang import Gang
from fairseq2.logging import get_log_writer
from fairseq2.models.seq2seq import Seq2SeqBatch
from fairseq2.models.sequence import SequenceBatch
from fairseq2.models.wav2vec2.asr import Wav2Vec2AsrModel, Wav2Vec2AsrOutput
from fairseq2.nn.utils.module import freeze_parameters
from fairseq2.recipes.criterion import AbstractCriterion
from fairseq2.recipes.wav2vec2.asr.metrics import (
    Wav2Vec2AsrMetricBag,
    Wav2Vec2AsrValidMetricBag,
)
from fairseq2.typing import override

log = get_log_writer(__name__)


@final
class Wav2Vec2AsrCriterion(AbstractCriterion[Seq2SeqBatch]):
    """Computes CTC loss using a wav2vec 2.0 ASR model."""

    _train_metric_bag: Wav2Vec2AsrMetricBag
    _valid_metric_bag: Wav2Vec2AsrValidMetricBag
    _freeze_encoder_for_n_steps: int

    def __init__(
        self,
        model: Module,
        gang: Gang,
        tokenizer: TextTokenizer,
        *,
        freeze_encoder_for_n_steps: int = 0,
        wer_file: Optional[Path] = None,
    ) -> None:
        """
        :param model:
            The wav2vec 2.0 ASR model. Can be DDP or FSDP wrapped.
        :param gang:
            The gang for distributed training or evaluation.
        :param tokenizer:
            The tokenizer to use.
        :param freeze_encoder_for_n_steps:
            The encoder will be frozen for this number of steps.
        :param wer_file:
            The output file to dump transcriptions, WER, and UER metrics.
        """
        super().__init__(model)

        self._train_metric_bag = Wav2Vec2AsrMetricBag(gang)

        self._valid_metric_bag = Wav2Vec2AsrValidMetricBag(
            gang, tokenizer, wer_file=wer_file
        )

        self._freeze_encoder_for_n_steps = freeze_encoder_for_n_steps

    @override
    def set_step(self, step_nr: int) -> None:
        if not self._model.training:
            return

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

    @override
    def compute_loss(self, batch: Seq2SeqBatch) -> Tuple[Tensor, int]:
        b = SequenceBatch(batch.source_seqs, batch.source_padding_mask)

        output = cast(Wav2Vec2AsrOutput, self._model(b))

        loss = output.compute_loss(batch.target_seqs, batch.target_padding_mask)

        if self._model.training:
            self._train_metric_bag.update(batch, loss.detach())
        else:
            self._valid_metric_bag.update(batch, loss.detach())
            self._valid_metric_bag.update_wer_metric(batch, output)

        return loss, batch.batch_size

    @final
    @property
    @override
    def train_metric_bag(self) -> Wav2Vec2AsrMetricBag:
        return self._train_metric_bag

    @final
    @property
    @override
    def valid_metric_bag(self) -> Wav2Vec2AsrValidMetricBag:
        return self._valid_metric_bag

    @final
    @property
    @override
    def throughput_metric_name(self) -> str:
        return "num_source_elements"
