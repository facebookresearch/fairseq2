# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple

import kenlm
import torch

import torch.nn.functional as F
from fairseq2.data.text.tokenizers import TextTokenizer

from fairseq2.models.seq2seq import Seq2SeqBatch
from fairseq2.models.sequence import SequenceBatch
from fairseq2.nn.padding import get_seq_lens, pad_seqs, PaddingMask

from torch import Tensor
from torch.nn import Module
from torch.nn.functional import ctc_loss, log_softmax


class AsrModel(Module, ABC):
    """Represents an Automatic Speech Recognition model."""

    @abstractmethod
    def forward(self, batch: SequenceBatch | Seq2SeqBatch) -> AsrModelOutput: ...


@dataclass
class CTCBeam:
    hyp: List[int]
    score: float = 0.0


def _get_ngram_score(
    ngram_model: kenlm.Model,
    tokenizer: TextTokenizer,
    parent_hyp: List[int],
    char_idx: int,
    blank_label: int = 0,
) -> float:
    assert char_idx != blank_label

    print("rescoring...")
    decoder = tokenizer.create_decoder()
    toks = torch.tensor(parent_hyp).unique_consecutive()
    toks = toks[toks != blank_label]

    hyp_parent = " ".join(decoder(torch.tensor(parent_hyp)))
    new_char = decoder(torch.tensor([char_idx]))
    hyp = hyp_parent + " " + new_char

    ngram_conditional_prob = ngram_model.score(
        hyp, bos=True, eos=False
    ) - ngram_model.score(hyp_parent, bos=True, eos=False)
    print("done rescoring!")
    return ngram_conditional_prob


@dataclass
class AsrModelOutput:
    logits: Tensor
    """The logits for next-step prediction. *Shape:* :math:`(N,S_{out},V)`,
    where :math:`N` is the batch size, :math:`S_{out}` is the output sequence
    length, and :math:`V` is the size of the vocabulary."""

    padding_mask: PaddingMask | None
    """The padding mask of :attr:`logits`. *Shape:* :math:`(N,S_{out})`, where
    :math:`N` is the batch size and :math:`S_{out}` is the output sequence
    length."""

    def compute_loss(
        self, targets: Tensor, target_padding_mask: PaddingMask | None
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
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
        # (N, S, V)
        lprobs = log_softmax(self.logits, dim=-1, dtype=torch.float32)

        # (N, S, V) -> (S, N, V)
        lprobs_t = lprobs.transpose(0, 1)

        # (N)
        seq_lens = get_seq_lens(lprobs, self.padding_mask)

        # (N)
        target_seq_lens = get_seq_lens(targets, target_padding_mask)

        # ()
        return (
            ctc_loss(
                lprobs_t,
                targets,
                seq_lens,
                target_seq_lens,
                reduction="sum",
                zero_infinity=True,
            ),
            {},
        )

    @torch.no_grad()
    def generate_hypotheses_ngram_fuse_single_row(
        self,
        pad_idx: int,
        seq_len: int,
        log_probs: Tensor,  # [S, V]
        ngram_model: kenlm.Model,
        tokenizer: TextTokenizer,
        blank_label: int = 0,
        nbest: int = 5,  # n beams
        topk: int = 100,  # only rescore top-k CTC candidates per beam
        rescore_beta: float = 0.1,
    ) -> Tensor:
        beams = [CTCBeam(hyp=[], score=0.0)] * nbest
        _, V = log_probs.size()

        for t in range(seq_len):
            new_beams = []
            _, cand_idxs_t = log_probs[t].topk(k=topk, dim=-1)

            for beam in beams:
                for v in range(V):
                    prfx = beam.hyp
                    if len(prfx) == 0:
                        seq = [v]
                    else:
                        seq = beam.hyp.append(v)
                    if v == blank_label or v == beam.hyp[-1]:
                        score = beam.score + log_probs[t, v]
                    elif v not in cand_idxs_t:
                        score = -torch.inf  # only rescore top-K candidates
                    else:
                        score = (
                            beam.score
                            + log_probs[t, v]
                            + rescore_beta
                            * _get_ngram_score(
                                ngram_model, tokenizer, beam.hyp, v, blank_label
                            )
                        )
                    # assert isinstance(seq, list)
                    # assert isinstance(score, float)
                    new_beams.append(CTCBeam(hyp=seq, score=score))  # type: ignore

            new_beams = sorted(new_beams, key=lambda x: x.score, reverse=True)[:nbest]
            beams = new_beams

        best_beam = beams[0]
        best_seq = torch.tensor(best_beam.hyp)
        best_seq = best_seq.unique_consecutive()
        best_seq = best_seq[best_seq != blank_label]
        return best_seq

    @torch.no_grad()
    def generate_hypotheses_ngram_fuse(
        self,
        pad_idx: int,
        tokenizer: TextTokenizer,
        ngram_model_dict: Dict[str, kenlm.Model],
        languages: List[str],  # list of languages [N]
        blank_label: int = 0,
        nbest: int = 5,
        rescore_beta: float = 0.1,
    ) -> tuple[Tensor, PaddingMask | None]:

        seq_lens = get_seq_lens(self.logits, self.padding_mask)
        log_probs = F.log_softmax(self.logits, dim=-1, dtype=torch.float32)  # (N, S, V)

        hyp_seq_list = []
        for language, log_prob, seq_len in zip(languages, log_probs, seq_lens):
            print("generating hypothesis using ngram fusion...")
            hyp_seq = self.generate_hypotheses_ngram_fuse_single_row(
                pad_idx,
                int(seq_len.item()),
                log_prob,
                ngram_model_dict[language],
                tokenizer,
                blank_label,
                nbest,
                rescore_beta=rescore_beta,
            )
            hyp_seq_list.append(hyp_seq)
            print("done generating hypothesis!")

        # (N, S), (N, S)
        return pad_seqs(hyp_seq_list, pad_value=pad_idx)

    def generate_hypotheses(
        self, pad_idx: int, blank_label: int = 0
    ) -> tuple[Tensor, PaddingMask | None]:
        """Generate hypotheses using greedy search.

        :param pad_idx:
            The index of the PAD symbol in the target vocabulary.
        :param blank_label:
            The blank label in logits.

        :returns:
            - The generated token (i.e. unit) sequences. *Shape:* :math:`(N,S)`,
              where :math:`N` is the batch size and :math:`S` is the sequence
              length.
            - The padding mask of the generated sequences. *Shape:* Same as the
              generated sequences.
        """
        seq_lens = get_seq_lens(self.logits, self.padding_mask)

        hyp_seq_list = []

        # Get the greedy token (i.e. unit) output of the model.
        for logits, seq_len in zip(self.logits, seq_lens):
            # (S)
            hyp_seq = logits[:seq_len].argmax(-1).unique_consecutive()

            # (S - blank)
            hyp_seq = hyp_seq[hyp_seq != blank_label]

            hyp_seq_list.append(hyp_seq)

        # (N, S), (N, S)
        return pad_seqs(hyp_seq_list, pad_value=pad_idx)
