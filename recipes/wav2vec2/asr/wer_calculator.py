# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import MutableMapping
from typing import TextIO, cast, final

from torch import Tensor

from fairseq2.data.tokenizers import TokenDecoder, Tokenizer
from fairseq2.datasets import Seq2SeqBatch
from fairseq2.error import InternalError, raise_operational_system_error
from fairseq2.file_system import FileMode
from fairseq2.metrics import MetricBag
from fairseq2.metrics.text import WerMetric
from fairseq2.nn import BatchLayout
from fairseq2.nn.utils.padding import pad_seqs
from fairseq2.recipe.base import RecipeContext


@final
class WerCalculator:
    """Computes Word Error Rate (WER) during validation/evaluation by comparing model predictions
    with ground truth transcriptions using greedy CTC decoding."""

    _text_decoder: TokenDecoder
    _pad_idx: int
    _blank_label: int
    _ref_output_stream: TextIO | None
    _hyp_output_stream: TextIO | None
    _wer_key: str = "wer"
    _uer_key: str = "uer"

    def __init__(
        self,
        tokenizer: Tokenizer,
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

    @classmethod
    def from_context(cls, context: RecipeContext) -> "WerCalculator":
        """Creates a WerCalculator by gluing output paths together for saving transcriptions. Only TP=0 and every DP rank run this."""

        if context.gangs.tp.rank == 0:
            file_system = context.file_system

            rank = context.gangs.dp.rank

            ref_file = context.output_dir.joinpath(
                f"transcriptions/rank_{rank}.ref.txt"
            )
            hyp_file = context.output_dir.joinpath(
                f"transcriptions/rank_{rank}.hyp.txt"
            )

            try:
                file_system.make_directory(ref_file.parent)
            except OSError as ex:
                raise_operational_system_error(ex)
            try:
                ref_fp = file_system.open_text(ref_file, mode=FileMode.WRITE)
            except OSError as ex:
                raise_operational_system_error(ex)
            try:
                hyp_fp = file_system.open_text(hyp_file, mode=FileMode.WRITE)
            except OSError as ex:
                raise_operational_system_error(ex)
        else:
            ref_fp = None
            hyp_fp = None

        return cls(
            tokenizer=context.tokenizer,
            ref_output_stream=ref_fp,
            hyp_output_stream=hyp_fp,
        )

    def compute_wer(
        self,
        batch: Seq2SeqBatch,
        logits: Tensor,
        logits_layout: BatchLayout,
        metric_bag: MetricBag,
    ) -> None:
        """Decode CTC logits to text and compute WER metrics.
        Performs greedy decoding (argmax + unique_consecutive) to generate hypotheses,
        then compares with reference transcriptions to compute WER and UER.
        """
        # (N, S)
        ref_seqs, ref_seqs_layout = batch.as_target_input()

        # (N, S)
        hyp_seqs, hyp_seqs_layout = self._generate_hypotheses(logits, logits_layout)

        refs = [self._text_decoder(s) for s in ref_seqs]
        hyps = [self._text_decoder(s) for s in hyp_seqs]

        metric_bag.get(WerMetric, self._wer_key).update(
            refs, ref_seqs, ref_seqs_layout, hyps, hyp_seqs, hyp_seqs_layout
        )

        try:
            # Write transcriptions if streams are provided
            if self._ref_output_stream is not None:
                for ref in refs:
                    self._ref_output_stream.write(ref + "\n")
                self._ref_output_stream.flush()

            if self._hyp_output_stream is not None:
                for hyp in hyps:
                    self._hyp_output_stream.write(hyp + "\n")
                self._hyp_output_stream.flush()
        except OSError as ex:
            raise InternalError(
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
        """TODO"""
        value = values.pop(self._wer_key, None)
        if value is not None:
            uer, wer = cast(tuple[Tensor, Tensor], value)
            if uer >= 1.0 and wer >= 1.0:
                values[self._uer_key] = uer
                values[self._wer_key] = wer
