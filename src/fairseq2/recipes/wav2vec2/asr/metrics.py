# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from typing import Any, Dict, Optional, TextIO

import torch
from torch import Tensor
from torcheval.metrics import Mean, Sum

from fairseq2.data.text import TextTokenDecoder, TextTokenizer
from fairseq2.gang import Gang
from fairseq2.metrics import MetricBag
from fairseq2.metrics.wer import WerMetric
from fairseq2.models.seq2seq import Seq2SeqBatch
from fairseq2.models.wav2vec2.asr.model import Wav2Vec2AsrOutput
from fairseq2.typing import override


class Wav2Vec2AsrMetricBag(MetricBag):
    """Holds the training metrics of a wav2vec 2.0 ASR model."""

    _ctc_loss: Mean
    _batch_size: Mean
    _elements_per_batch: Mean
    _num_examples: Sum
    _num_source_elements: Sum
    _num_target_elements: Sum
    _total_num_examples: Sum
    _total_num_source_elements: Sum
    _total_num_target_elements: Sum

    def __init__(self, gang: Gang) -> None:
        """
        :param gang:
            The gang over which to sync metrics.
        """
        super().__init__(gang)

        d = gang.device

        self.register_metric("_ctc_loss", Mean(device=d), persistent=False)

        self.register_metric("_batch_size", Mean(device=d), persistent=False)

        self.register_metric("_elements_per_batch", Mean(device=d), persistent=False)

        self.register_metric("_num_examples", Sum(device=d), persistent=False)

        self.register_metric("_num_source_elements", Sum(device=d), persistent=False)
        self.register_metric("_num_target_elements", Sum(device=d), persistent=False)

        self._total_num_examples = Sum(device=d)

        self._total_num_source_elements = Sum(device=d)
        self._total_num_target_elements = Sum(device=d)

    @torch.inference_mode()
    def update_loss_metrics(self, batch: Seq2SeqBatch, ctc_loss: Tensor) -> None:
        """Update the loss metrics.

        :param batch:
            The batch processed by the model.
        :param ctc_loss:
            The loss of ``batch``.
        """
        batch_size = torch.tensor(batch.batch_size)

        num_source_elements = torch.tensor(batch.num_source_elements())
        num_target_elements = torch.tensor(batch.num_target_elements())

        normalized_ctc_loss = ctc_loss.cpu() / batch.batch_size / math.log(2)

        self._ctc_loss.update(normalized_ctc_loss, weight=batch_size)

        self._batch_size.update(batch_size * self._gang.size)

        self._elements_per_batch.update(num_source_elements * self._gang.size)

        self._num_examples.update(batch_size)

        self._num_source_elements.update(num_source_elements)
        self._num_target_elements.update(num_target_elements)

        self._total_num_examples.update(batch_size)

        self._total_num_source_elements.update(num_source_elements)
        self._total_num_target_elements.update(num_target_elements)


class Wav2Vec2AsrEvalMetricBag(Wav2Vec2AsrMetricBag):
    """Holds the evaluation metrics of a wav2vec 2.0 ASR model."""

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
            The text tokenizer to compute the WER (Word Error Rate).
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
        """Update the WER (Word Error Rate) metric.

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
