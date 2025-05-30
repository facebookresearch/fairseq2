# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import MutableMapping
from typing import TextIO, cast, final

from torch import Tensor

from fairseq2.data.text.tokenizers import TextTokenDecoder, TextTokenizer
from fairseq2.datasets import Seq2SeqBatch
from fairseq2.metrics import MetricBag
from fairseq2.metrics.text import WerMetric
from fairseq2.nn import BatchLayout
from fairseq2.nn.utils.padding import pad_seqs
from fairseq2.recipes import UnitError


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
        self._text_decoder = tokenizer.create_decoder(skip_special_tokens=True)

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
        self,
        batch: Seq2SeqBatch,
        logits: Tensor,
        logits_layout: BatchLayout,
        metric_bag: MetricBag,
    ) -> None:
        # (N, S)
        ref_seqs, ref_seqs_layout = batch.as_target_input()

        # (N, S)
        hyp_seqs, hyp_seqs_layout = self._generate_hypotheses(logits, logits_layout)

        refs = [self._text_decoder(s) for s in ref_seqs]
        hyps = [self._text_decoder(s) for s in hyp_seqs]

        metric_bag.get(WerMetric, "wer").update(
            refs, ref_seqs, ref_seqs_layout, hyps, hyp_seqs, hyp_seqs_layout
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

    def _generate_hypotheses(
        self, logits: Tensor, logits_layout: BatchLayout
    ) -> tuple[Tensor, BatchLayout]:
        hyp_seqs = []

        # Get the greedy token (i.e. unit) output of the model.
        for logits, logits_len in zip(logits, logits_layout.seq_lens):
            # (S)
            hyp_seq = logits[:logits_len].argmax(-1).unique_consecutive()

            # (S - blank)
            hyp_seq = hyp_seq[hyp_seq != self._blank_label]

            hyp_seqs.append(hyp_seq)

        # (N, S), (N, S)
        return pad_seqs(hyp_seqs, pad_value=self._pad_idx)

    def process_metric_values(self, values: MutableMapping[str, object]) -> None:
        value = values.pop("wer", None)
        if value is not None:
            uer, wer = cast(tuple[Tensor, Tensor], value)

            if uer >= 1.0 and wer >= 1.0:
                values["uer"] = uer
                values["wer"] = wer
