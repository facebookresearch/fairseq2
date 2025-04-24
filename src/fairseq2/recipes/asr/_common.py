# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from typing import Any, final, TextIO

import torch

from fairseq2.data.text.tokenizers import TextTokenDecoder, TextTokenizer
from fairseq2.gang import Gang
from fairseq2.metrics import Mean
from fairseq2.metrics.text import WerMetric
from fairseq2.models.asr import AsrModel, AsrModelOutput
from fairseq2.models.seq2seq import Seq2SeqBatch
from fairseq2.models.sequence import SequenceBatch

from fairseq2.models.wav2vec2.asr import Wav2Vec2AsrModel
from fairseq2.recipes import BaseMetricBag, Model, UnitError
from torch import Tensor
from typing_extensions import override


@final
class AsrCriterion:
    _model: Model
    _scorer: AsrScorer | None

    def __init__(self, model: Model, scorer: AsrScorer | None = None) -> None:
        if not isinstance(model.base_module, AsrModel):
            raise TypeError(
                f"`model.base_module` must be of type `{AsrModel}`, but is of type `{type(model.base_module)}` instead."
            )

        self._model = model

        self._scorer = scorer

    def __call__(
        self, batch: Seq2SeqBatch, metric_bag: AsrMetricBag
    ) -> tuple[Tensor, int]:
        if isinstance(self._model, Wav2Vec2AsrModel):
            input_batch = SequenceBatch(batch.source_seqs, batch.source_padding_mask)
        else:
            input_batch = batch

        output = self._forward(input_batch)

        loss = output.compute_loss(batch.target_seqs, batch.target_padding_mask)

        metric_bag.update_ctc_loss(batch, loss)

        metric_bag.update_batch_metrics(batch)

        if self._scorer is not None:
            self._scorer(batch, output, metric_bag)

        return loss, batch.batch_size

    def _forward(self, batch: SequenceBatch | Seq2SeqBatch) -> AsrModelOutput:
        return self._model.module(batch)  # type: ignore[no-any-return]

    @property
    def model(self) -> Model:
        return self._model


@final
class AsrScorer:
    _text_decoder: TextTokenDecoder
    _pad_idx: int
    _blank_label: int
    _ref_output_stream: TextIO | None
    _hyp_output_stream: TextIO | None

    def __init__(
        self,
        tokenizer: TextTokenizer,
        *,
        blank_label: int = 0,
        ref_output_stream: TextIO | None = None,
        hyp_output_stream: TextIO | None = None,
    ) -> None:
        """
        :param tokenizer: The tokenizer to encode target text.
        :param blank_label: The blank label in logits.
        :param ref_output_stream: The output stream to dump references.
        :param hyp_output_stream: The output stream to dump hypotheses.
        """
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

    def __call__(
        self, batch: Seq2SeqBatch, output: AsrModelOutput, metric_bag: AsrMetricBag
    ) -> None:
        # (N, S), (N, S)
        ref_seqs, ref_padding_mask = batch.target_seqs, batch.target_padding_mask

        # (N, S), (N, S)
        hyp_seqs, hyp_padding_mask = output.generate_hypotheses(
            self._pad_idx, self._blank_label
        )

        refs = [self._text_decoder(s) for s in ref_seqs]
        hyps = [self._text_decoder(s) for s in hyp_seqs]

        metric_bag.wer.update(
            refs, ref_seqs, ref_padding_mask, hyps, hyp_seqs, hyp_padding_mask
        )

        try:
            # Dump references.
            stream = self._ref_output_stream
            if stream is not None:
                for ref in refs:
                    stream.write(ref)
                    stream.write("\n")

                stream.flush()

            # Dump hypotheses.
            stream = self._hyp_output_stream
            if stream is not None:
                for hyp in hyps:
                    stream.write(hyp)
                    stream.write("\n")

                stream.flush()
        except OSError as ex:
            raise UnitError(
                "The generator output cannot be written. See the nested exception for details."
            ) from ex


class AsrMetricBag(BaseMetricBag):
    ctc_loss: Mean
    wer: WerMetric

    def __init__(self, gang: Gang, train: bool = True) -> None:
        super().__init__(gang, train=train)

        d = gang.device

        self.register_metric("ctc_loss", Mean(device=d), persistent=False)

        self.register_metric("wer", WerMetric(device=d), persistent=False)

    @torch.inference_mode()
    def update_ctc_loss(self, batch: Seq2SeqBatch, loss: Tensor) -> None:
        n = batch.batch_size

        self.ctc_loss.update(loss.detach() / n / math.log(2), weight=n)

    @torch.inference_mode()
    def update_batch_metrics(self, batch: Seq2SeqBatch) -> None:
        num_examples = batch.batch_size

        num_elements = batch.num_source_elements()

        self.num_examples.update(num_examples)
        self.num_elements.update(num_elements)

        if self._train:
            assert self.total_num_examples is not None
            assert self.total_num_elements is not None

            self.total_num_examples.update(num_examples)
            self.total_num_elements.update(num_elements)

    @override
    def process_metric_values(self, values: dict[str, Any]) -> None:
        super().process_metric_values(values)

        uer, wer = values.pop("wer")

        if uer >= 0.0 and wer >= 0.0:
            values["uer"] = uer
            values["wer"] = wer
