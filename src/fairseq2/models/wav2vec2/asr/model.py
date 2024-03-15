# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, final

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Dropout
from torch.nn.functional import ctc_loss, log_softmax
from torcheval.metrics import Mean, Sum, Throughput

from fairseq2.data.text import TextTokenizer
from fairseq2.gang import Gang
from fairseq2.metrics import MetricBag, WerMetric
from fairseq2.models.model import Model
from fairseq2.models.seq2seq import Seq2SeqBatch
from fairseq2.models.wav2vec2.frontend import Wav2Vec2Frontend
from fairseq2.models.wav2vec2.masker import Wav2Vec2Masker
from fairseq2.nn import Linear
from fairseq2.nn.padding import PaddingMask, get_seq_lens, pad_seqs
from fairseq2.nn.transformer import TransformerEncoder
from fairseq2.typing import DataType, Device, override
from fairseq2.utils.profiler import Stopwatch


@final
class Wav2Vec2AsrModel(Model):
    """Represents a wav2vec 2.0 ASR model as described in
    :cite:t:`https://doi.org/10.48550/arxiv.2006.11477`."""

    model_dim: int
    encoder_frontend: Wav2Vec2Frontend
    masker: Optional[Wav2Vec2Masker]
    final_dropout: Optional[Dropout]
    final_proj: Linear

    def __init__(
        self,
        encoder_frontend: Wav2Vec2Frontend,
        encoder: TransformerEncoder,
        final_dim: int,
        *,
        masker: Optional[Wav2Vec2Masker] = None,
        final_dropout_p: float = 0.0,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param encoder_frontend:
            The encoder frontend.
        :param encoder:
            The encoder (i.e. context network).
        :param final_dim:
            The dimensionality of the final projection that is applied to
            context network outputs.
        :param masker:
            The feature masker.
        :param final_dropout_p:
            The dropout probability on context network outputs.
        """
        super().__init__()

        self.model_dim = encoder.model_dim

        self.encoder_frontend = encoder_frontend
        self.encoder = encoder

        self.register_module("masker", masker)

        if final_dropout_p > 0.0:
            self.final_dropout = Dropout(final_dropout_p)
        else:
            self.register_module("final_dropout", None)

        self.final_proj = Linear(
            self.model_dim,
            final_dim,
            bias=True,
            init_fn=init_final_projection,
            device=device,
            dtype=dtype,
        )

    def forward(self, batch: Seq2SeqBatch) -> Wav2Vec2AsrOutput:
        """
        :param batch:
            The batch of sequences to process.
        """
        seqs, padding_mask = self.encoder_frontend.extract_features(
            batch.source_seqs, batch.source_padding_mask
        )

        seqs, padding_mask, _ = self.encoder_frontend.process_features(
            seqs, padding_mask, self.masker if self.training else None
        )

        seqs, padding_mask = self.encoder(seqs, padding_mask)

        if self.final_dropout is not None:
            seqs = self.final_dropout(seqs)

        logits = self.final_proj(seqs)

        return Wav2Vec2AsrOutput(logits, padding_mask)


def init_final_projection(proj: Linear) -> None:
    """Initialize ``proj`` as the final projection of a wav2vec 2.0 ASR model."""
    nn.init.xavier_uniform_(proj.weight)

    if proj.bias is not None:
        nn.init.zeros_(proj.bias)


@final
@dataclass
class Wav2Vec2AsrOutput:
    logits: Tensor
    """The logits for next-step prediction. *Shape:* :math:`(N,S_{out},T)`,
    where :math:`N` is the batch size, :math:`S_{out}` is the output sequence
    length, and :math:`T` is the size of the vocabulary."""

    padding_mask: Optional[PaddingMask]
    """The padding mask of :attr:`logits`. *Shape:* :math:`(N,S_{out})`, where
    :math:`N` is the batch size and :math:`S_{out}` is the output sequence
    length."""

    def compute_loss(
        self, targets: Tensor, target_padding_mask: Optional[PaddingMask]
    ) -> Tensor:
        """Compute the CTC (Connectionist Temporal Classification) loss.

        :param targets:
            The target indices. *Shape:* :math:`(N,S_{tgt})`, where :math:`N` is
            the batch size and :math:`S_{tgt}` is the target sequence length.
        :param target_padding_mask:
            The padding mask of the targets. *Shape:* Same as ``targets``.

        :returns:
            A scalar tensor representing the summed CTC loss.
        """
        # For numerical stability run in single precision.
        # (N, S, T)
        lprobs = log_softmax(self.logits, dim=-1, dtype=torch.float32)

        # (N, S, T) -> (S, N, T)
        lprobs_t = lprobs.transpose(0, 1)

        # (N)
        seq_lens = get_seq_lens(lprobs, self.padding_mask)

        # (N)
        target_seq_lens = get_seq_lens(targets, target_padding_mask)

        # ()
        return ctc_loss(
            lprobs_t,
            targets,
            seq_lens,
            target_seq_lens,
            reduction="sum",
            zero_infinity=True,
        )


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
    _blank_idx: int

    def __init__(
        self,
        gang: Gang,
        tokenizer: TextTokenizer,
        *,
        blank_idx: int = 0,
        wall_time: Optional[Stopwatch] = None,
    ) -> None:
        """
        :param gang:
            The gang to sync metrics across all processes.
        :param tokenizer:
            The text tokenizer to compute the WER (Word Error Rate).
        :param blank_idx:
            The index of the blank symbol.
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

        self._blank_idx = blank_idx

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
        seq_lens = get_seq_lens(model_output.logits, model_output.padding_mask)

        hyp_seq_list = []

        # Get the greedy token (i.e. unit) output of the model.
        for logits, seq_len in zip(model_output.logits, seq_lens):
            # (S)
            hyp_seq = logits[:seq_len].argmax(-1).unique_consecutive()

            # (S - blank)
            hyp_seq = hyp_seq[hyp_seq != self._blank_idx]

            hyp_seq_list.append(hyp_seq)

        # (N, S), (N, S)
        hyp_seqs, hyp_padding_mask = pad_seqs(hyp_seq_list, pad_value=self._pad_idx)

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
