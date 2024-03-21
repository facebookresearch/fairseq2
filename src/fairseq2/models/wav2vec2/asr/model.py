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

from fairseq2.gang import Gang
from fairseq2.metrics import MetricBag
from fairseq2.models.model import Model
from fairseq2.models.seq2seq import Seq2SeqBatch
from fairseq2.models.wav2vec2.frontend import Wav2Vec2Frontend
from fairseq2.models.wav2vec2.masker import Wav2Vec2Masker
from fairseq2.nn import Linear
from fairseq2.nn.padding import PaddingMask, get_seq_lens
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

        encoder_output, padding_mask = self.encoder(seqs, padding_mask)

        seqs = encoder_output

        if self.final_dropout is not None:
            seqs = self.final_dropout(seqs)

        logits = self.final_proj(seqs)

        return Wav2Vec2AsrOutput(logits, encoder_output, padding_mask)


def init_final_projection(proj: Linear) -> None:
    """Initialize ``proj`` as the final projection of a wav2vec 2.0 ASR model."""
    nn.init.xavier_uniform_(proj.weight)

    if proj.bias is not None:
        nn.init.zeros_(proj.bias)


@dataclass
class Wav2Vec2AsrLoss:
    """Holds the loss and lprobs of a wav2vec 2.0 ASR model."""
    loss: Tensor

    lprobs: Tensor


@dataclass
class Wav2Vec2AsrOutput:
    logits: Tensor
    """The logits for next-step prediction. *Shape:* :math:`(N,S,T)`, where
    :math:`N` is the batch size, :math:`S` is the source sequence length, and
    :math:`T` is the size of the vocabulary."""

    encoder_output: Tensor
    """The encoder output. *Shape:* :math:`(N,S_{enc},M)`, where :math:`N` is
    the batch size, :math:`S_{enc}` is the encoder output sequence length, and
    :math:`M` is the dimensionality of the model."""

    encoder_padding_mask: Optional[PaddingMask]
    """The padding mask of the encoder output. *Shape:* :math:`(N,S_{enc})`,
    where :math:`N` is the batch size and :math:`S_{enc}` is the encoder output
    sequence length."""

    def compute_loss(
        self, targets: Tensor, target_padding_mask: Optional[PaddingMask]
    ) -> Wav2Vec2AsrLoss:
        """Compute the CTC (Connectionist Temporal Classification) loss.

        :param targets:
            The target indices. *Shape:* :math:`(N,S)`, where :math:`N` is the
            batch size and :math:`S` is the target sequence length.
        :param target_padding_mask:
            The padding mask of the targets. *Shape:* Same as ``targets``.

        :returns:
            A scalar tensor representing the summed CTC loss.
        """
        # For numerical stability run in single precision.
        # (N, S, T)
        lprobs = log_softmax(self.logits, dim=-1, dtype=torch.float32)

        # (N, S, T) -> (S, N, T)
        lprobs = lprobs.transpose(0, 1)

        # (N)
        target_seq_lens = get_seq_lens(targets, target_padding_mask)

        # (N)
        feature_seq_lens = get_seq_lens(self.encoder_output, self.encoder_padding_mask)

        # ()
        loss = ctc_loss(
            lprobs,
            targets,
            feature_seq_lens,
            target_seq_lens,
            reduction="sum",
            zero_infinity=True,
        )

        # (S, N, T) -> (N, S, T)
        lprobs = lprobs.transpose(0, 1)

        return Wav2Vec2AsrLoss(loss, lprobs)


class Wav2Vec2AsrMetricBag(MetricBag):
    """Holds the common metrics of a wav2vec 2.0 ASR model."""

    ctc_loss: Mean
    batch_size: Mean
    elements_per_batch: Mean
    elements_per_second: Throughput
    num_examples: Sum
    num_source_elements: Sum
    num_target_elements: Sum

    def __init__(self, gang: Gang, wall_time: Optional[Stopwatch] = None) -> None:
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

    def update_metrics(
        self,
        batches: Sequence[Seq2SeqBatch],
        ctc_losses: Sequence[Tensor],
        elapsed_time: float,
    ) -> None:
        """Update the metrics.

        :param batches:
            The batches processed by the model in the last training step.
        :param ctc_losses:
            The CTC losses generated by the model for each batch in ``batches``.
        :param elapsed_time:
            The total elapsed time to read and process ``batches``.
        """
        ctc_loss = torch.zeros((), dtype=torch.float64)

        batch_size = torch.zeros((), dtype=torch.float64)

        num_source_elements = torch.zeros((), dtype=torch.float64)
        num_target_elements = torch.zeros((), dtype=torch.float64)

        for batch, batch_ctc_loss in zip(batches, ctc_losses):
            ctc_loss += float(batch_ctc_loss)

            batch_size += batch.batch_size

            num_source_elements += batch.num_source_elements()
            num_target_elements += batch.num_target_elements() - batch.batch_size

        self.ctc_loss.update(ctc_loss / batch_size / math.log(2), weight=batch_size)

        self.batch_size.update(batch_size * self._gang.size)

        self.elements_per_batch.update(num_source_elements * self._gang.size)

        self.elements_per_second.update(int(num_source_elements), elapsed_time)

        self.num_examples.update(batch_size)

        self.num_source_elements.update(num_source_elements)
        self.num_target_elements.update(num_target_elements)

    def reset_batch_metrics(self) -> None:
        """Reset the batch metrics to their initial state."""
        self.ctc_loss.reset()
        self.batch_size.reset()
        self.elements_per_batch.reset()
        self.elements_per_second.reset()

    @override
    def process_metric_values(self, values: Dict[str, Any]) -> None:
        super().process_metric_values(values)

        values["elapsed_time"] = self.elements_per_second.elapsed_time_sec
