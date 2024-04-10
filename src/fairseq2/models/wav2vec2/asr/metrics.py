# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence

import torch
from torch import Tensor
from torcheval.metrics import Mean, Sum, Throughput

from fairseq2.data.text import TextTokenizer
from fairseq2.gang import Gang
from fairseq2.metrics import MetricBag
from fairseq2.metrics.wer_metric import WerMetric
from fairseq2.models.seq2seq import Seq2SeqBatch
from fairseq2.models.wav2vec2.asr.model import Wav2Vec2AsrOutput
from fairseq2.typing import override
from fairseq2.utils.profiler import Stopwatch


class Wav2Vec2AsrMetricBag(MetricBag):
    """Holds the common metrics of a wav2vec 2.0 ASR model."""

    ctc_loss: Mean
    batch_size: Mean
    elements_per_batch: Mean
    elements_per_second: Throughput
    num_examples: Sum
    num_source_elements: Sum
    num_target_elements: Sum

    def __init__(self, gang: Gang, *, wall_time: Optional[Stopwatch] = None) -> None:
        """
        :param gang:
            The gang to sync metrics across all processes.
        :param wall_time:
            The :class:`Stopwatch` to keep track of process wall time.
        """
        super().__init__(gang, wall_time)

        d = gang.device

        self.register_metric("ctc_loss", Mean(device=d), persistent=False)

        self.register_metric("batch_size", Mean(device=d), persistent=False)

        self.register_metric("elements_per_batch", Mean(device=d), persistent=False)

        self.register_metric(
            "elements_per_second", Throughput(device=d), persistent=False
        )

        self.num_examples = Sum(device=d)

        self.num_source_elements = Sum(device=d)
        self.num_target_elements = Sum(device=d)

    @torch.inference_mode()
    def update_step_metrics(
        self,
        batches: Sequence[Seq2SeqBatch],
        ctc_losses: Sequence[Tensor],
        time: Stopwatch,
    ) -> None:
        """Update the step metrics.

        :param batches:
            The batches processed by the model.
        :param ctc_losses:
            The CTC losses output by the model for ``batches``.
        :param time:
            The :class:`Stopwatch` to keep track of elapsed time.
        """
        ctc_loss = torch.zeros((), dtype=torch.float64)

        batch_size = torch.zeros((), dtype=torch.float64)

        num_source_elements = torch.zeros((), dtype=torch.float64)
        num_target_elements = torch.zeros((), dtype=torch.float64)

        for batch, batch_ctc_loss in zip(batches, ctc_losses):
            ctc_loss += float(batch_ctc_loss)

            batch_size += batch.batch_size

            num_source_elements += batch.num_source_elements()
            num_target_elements += batch.num_target_elements()

        self.ctc_loss.update(ctc_loss / batch_size / math.log(2), weight=batch_size)

        self.batch_size.update(batch_size * self._gang.size)

        self.num_examples.update(batch_size)

        self.num_source_elements.update(num_source_elements)
        self.num_target_elements.update(num_target_elements)

        self.elements_per_batch.update(num_source_elements * self._gang.size)

        self.elements_per_second.update(
            int(num_source_elements), time.get_elapsed_time()
        )

    def reset_step_metrics(self) -> None:
        """Reset the step metrics to their initial state."""
        self.ctc_loss.reset()
        self.batch_size.reset()
        self.elements_per_batch.reset()
        self.elements_per_second.reset()

    @override
    def process_metric_values(self, values: Dict[str, Any]) -> None:
        super().process_metric_values(values)

        values["elapsed_time"] = self.elements_per_second.elapsed_time_sec


class Wav2Vec2AsrValidMetricBag(Wav2Vec2AsrMetricBag):
    """Holds the common validation metrics of a wav2vec 2.0 ASR model."""

    wer: WerMetric

    _pad_idx: int
    _blank_label: int

    def __init__(
        self,
        gang: Gang,
        tokenizer: TextTokenizer,
        *,
        blank_label: int = 0,
        wall_time: Optional[Stopwatch] = None,
    ) -> None:
        """
        :param gang:
            The gang to sync metrics across all processes.
        :param tokenizer:
            The text tokenizer to compute the WER (Word Error Rate).
        :param blank_label:
            The blank label in logits.
        :param wall_time:
            The :class:`Stopwatch` to keep track of process wall time.
        """
        super().__init__(gang, wall_time=wall_time)

        wer = WerMetric(tokenizer=tokenizer, device=gang.device)

        self.register_metric("wer", wer, persistent=False)

        pad_idx = tokenizer.vocab_info.pad_idx
        if pad_idx is None:
            raise ValueError(
                "``vocab_info` of `tokenizer` must have a PAD symbol defined."
            )

        self._pad_idx = pad_idx

        self._blank_label = blank_label

    @torch.inference_mode()
    def update_wer_metric(
        self, batch: Seq2SeqBatch, model_output: Wav2Vec2AsrOutput
    ) -> None:
        """Update the WER (Word Error Rate).

        :param batch:
            The batch processed by the model.
        :param model_output:
            The output of the model for ``batch``.
        """
        # (N, S), (N, S)
        hyp_seqs, hyp_padding_mask = model_output.generate_hypotheses(
            self._pad_idx, self._blank_label
        )

        self.wer.update(
            batch.target_seqs,
            batch.target_padding_mask,
            hyp_seqs,
            hyp_padding_mask,
        )

    @override
    def reset_step_metrics(self) -> None:
        super().reset_step_metrics()

        self.wer.reset()

    @override
    def process_metric_values(self, values: Dict[str, Any]) -> None:
        super().process_metric_values(values)

        uer, wer = values.pop("wer")

        values["uer"] = uer
        values["wer"] = wer
