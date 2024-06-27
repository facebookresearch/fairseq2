# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, TextIO, final

from torch.nn import Module

from fairseq2.data.text import TextTokenizer
from fairseq2.gang import Gang
from fairseq2.models.seq2seq import Seq2SeqBatch
from fairseq2.models.sequence import SequenceBatch
from fairseq2.models.wav2vec2.asr import Wav2Vec2AsrModel, Wav2Vec2AsrOutput
from fairseq2.recipes.evaluator import AbstractEvalUnit
from fairseq2.recipes.utils.setup import check_model_type
from fairseq2.recipes.wav2vec2.asr.metrics import Wav2Vec2AsrEvalMetricBag
from fairseq2.typing import override


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
