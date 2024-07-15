# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple, final, Dict

import torch
from torch import Tensor
from torch.nn.functional import log_softmax
import torch.nn as nn
import torch.nn.functional as F
from fairseq2.data import VocabularyInfo
from fairseq2.models.model import Model
from fairseq2.nn.functional import nll_loss
from fairseq2.nn.padding import PaddingMask
import editdistance

class SequenceModel(Model, ABC):
    """Represents a sequence model."""

    max_seq_len: int
    vocab_info: VocabularyInfo

    def __init__(self, max_seq_len: int, vocab_info: VocabularyInfo) -> None:
        """
        :param max_seq_len:
            The maximum length of sequences produced by the model.
        :param vocab_info:
            The vocabulary information of sequences produced by the model.
        """
        super().__init__()

        self.max_seq_len = max_seq_len
        self.vocab_info = vocab_info

    @abstractmethod
    def forward(self, batch: SequenceBatch) -> SequenceModelOutput:
        """
        :param batch:
            The batch of sequences to process.
        """


@final
@dataclass
class SequenceBatch:
    """Represents a sequence batch."""

    seqs: Tensor
    """The sequences. *Shape:* :math:`(N,S,*)`, where :math:`N` is the batch
    size, :math:`S` is the sequence length, and :math:`*` is any number of
    sequence-specific dimensions including none."""

    padding_mask: Optional[PaddingMask]
    """The padding mask of :attr:`seqs`. *Shape:* :math:`(N,S)`, where :math:`N`
    is the batch size and :math:`S` is the sequence length."""

    target_mask: Optional[Tensor] = None
    """The mask specifying the elements in ``seqs`` that should be treated as
    targets during model training or validation. *Shape:* :math:`(N,S)`, where
    :math:`N` is the batch size and :math:`S` is the sequence length."""

    example: Any = None
    """The data example from which this batch was constructed."""

    @property
    def batch_size(self) -> int:
        """The size of the batch dimension."""
        return self.seqs.size(0)

    def num_elements(self) -> int:
        """Return the number of elements in the batch."""
        if self.padding_mask is None:
            return self.seqs.numel()

        return int(self.padding_mask.seq_lens.sum())

    def num_target_elements(self) -> int:
        """Return the number of target elements in the batch."""
        if self.target_mask is not None:
            return int(self.target_mask.sum())

        return self.num_elements()


def as_auto_regressive_input(
    batch: SequenceBatch,
) -> Tuple[SequenceBatch, SequenceBatch]:
    """Use ``batch`` to train an auto-regressive model.

    :returns:
        The tuple of input and target batches.
    """
    if (seq_len := batch.seqs.size(1)) < 2:
        raise ValueError(
            f"The sequence length of `batch.seqs` must be at least 2 for training, but is {seq_len} instead."
        )

    seqs, targets = batch.seqs[:, :-1], batch.seqs[:, 1:]

    if batch.padding_mask is None:
        padding_mask = None
    else:
        padding_mask = batch.padding_mask.trim(1)

    if batch.target_mask is None:
        seqs_target_mask, target_mask = None, None
    else:
        seqs_target_mask, target_mask = (
            batch.target_mask[:, :-1], batch.target_mask[:, 1:]  # fmt: skip
        )

    batch = SequenceBatch(seqs, padding_mask, seqs_target_mask, batch.example)

    target_batch = SequenceBatch(targets, padding_mask, target_mask)

    return batch, target_batch


# compat
@dataclass
class BCVocabInfo:
    pad_idx: Optional[int] = None


@final
@dataclass
class SequenceModelOutput:
    """Holds the output of a sequence model."""

    logits: Tensor
    """The logits for next-step prediction. *Shape:* :math:`(N,S,T)`, where
    :math:`N` is the batch size, :math:`S` is the sequence length, and :math:`T`
    is the size of the vocabulary."""

    pad_idx: Optional[int]
    """The index of the PAD symbols in the vocabulary."""

    # compat
    vocab_info: BCVocabInfo = field(default_factory=BCVocabInfo)

    # compat
    def __post_init__(self) -> None:
        self.vocab_info.pad_idx = self.pad_idx

    def compute_loss(
        self,
        targets: Tensor,
        *,
        loss_mask: Optional[Tensor] = None,
        ignore_prefix_size: int = 0,
        label_smoothing: float = 0.0,
    ) -> Tensor:
        """Compute the NLL (negative log-likelihood) loss.

        :param targets:
            The target indices. *Shape:* :math:`(N,S)`, where :math:`N` is the
            batch size and :math:`S` is the sequence length.
        :param loss_mask:
            The loss mask that specifies the elements in ``targets`` that should
            be used in the loss computation. All non-masked elements will be
            ignored. *Shape:* Same as ``targets``.
        :param ignore_prefix_size:
            The number of steps from the beginning of the sequence that should
            be ignored in the loss computation.
        :param label_smoothing:
            The amount of label smoothing to apply while computing the loss.

        :returns:
            A scalar tensor representing the summed NLL loss.
        """
        if ignore_prefix_size > 0:
            logits = self.logits[:, ignore_prefix_size:, :]
        else:
            logits = self.logits

        if ignore_prefix_size > 0:
            targets = targets[:, ignore_prefix_size:]

        # For numerical stability run in single precision.
        # (N, S, T)
        lprobs = log_softmax(logits, dim=-1, dtype=torch.float32)

        # sum: (), none: (N, S)
        loss = nll_loss(
            lprobs,
            targets,
            self.pad_idx,
            label_smoothing=label_smoothing,
            reduction="sum" if loss_mask is None else "none",
        )

        if loss_mask is None:
            return loss

        if ignore_prefix_size > 0:
            loss_mask = loss_mask[:, ignore_prefix_size:]

        # ()
        return (loss * loss_mask).sum()





@final
@dataclass
class SpeechTextReprOutput:
    speech_repr: Tensor
    text_repr: Tensor
    mask: PaddingMask
    mse_loss_fn = nn.MSELoss(reduce=None, reduction="none")

    def compute_loss(self, embed_table=None, text_tokens=None, compute_acc=False) -> Dict[Tensor]:
        mse_loss = self.mse_loss_fn(self.speech_repr, self.text_repr)
        if self.mask is not None:
            target_mask = self.mask.materialize()
        else:
            # no padding for the text tokens
            target_mask = torch.ones((self.speech_repr.shape[0], self.speech_repr.shape[1]), device=self.speech_repr.device).bool()
        # sum over embed dimension and avg on token level
        mse_loss = torch.sum(mse_loss[target_mask], dim=1).mean()
        # hid_dim = self.speech_repr.shape[2]
        norm_speech_repr = F.normalize(self.speech_repr, dim=2, p=2)[target_mask]
        norm_text_repr = F.normalize(self.text_repr, dim=2, p=2)[target_mask]
        cosine_sim = F.cosine_similarity(norm_speech_repr, norm_text_repr, dim=1)
        cosine_sim_loss = (1 - cosine_sim).mean()
        
        num_matches=None
        # check if best matches is the corresponding index
        num_elements = target_mask.sum().item()
        if compute_acc:
            assert embed_table is not None
            assert text_tokens is not None
            "embed: (vocab, dim), text_tokens: (bz, seq_len), norm_speech_repr: (valid_entry, embed_dim)"
            norm_embed = F.normalize(embed_table, p=2, dim=1)
            # (entries, dim) x (dim, vocab 0) --> (entries, vocab)
            text_sim = norm_speech_repr @ (norm_embed.T)
            closest_indices = torch.argmax(text_sim, dim=1)
            target_toks = text_tokens[target_mask]
            num_matches = (closest_indices == target_toks).sum()


        acc = (num_matches / num_elements).item() if compute_acc else None
        out = {
            "mse_loss": mse_loss, 
            "cosine_sim_loss": cosine_sim_loss, 
            "acc": acc,
            "target_size": num_elements
        }
        return out
    
    def compute_asr(self, embed_table=None, text_tokens=None, tokenizer=None) -> Dict[Tensor]:
        if self.mask is not None:
            target_mask = self.mask.materialize()
        else:
            # no padding for the text tokens
            target_mask = torch.ones((text_tokens.shape[0], text_tokens.shape[1]), device=text_tokens.device).bool()
        target_lens = target_mask.sum(dim=1)
        norm_speech_repr = F.normalize(self.speech_repr, dim=2, p=2)
        norm_embed = F.normalize(embed_table, p=2, dim=1)
        # (bz, seq_len, dim) x (dim, vocab 0) --> (bz, seq_len, vocab)
        text_sim = norm_speech_repr @ (norm_embed.T)
        pred_indices = torch.argmax(text_sim, dim=2)

        word_len, all_wer = 0, 0
        for i in range(text_tokens.shape[0]):
            ref_text = tokenizer(text_tokens[i, :target_lens[i]])
            ref_words = ref_text.split()
            pred_text = tokenizer(pred_indices[i, :target_lens[i]])
            pred_words = pred_text.split()
            wer =  editdistance.eval(pred_words, ref_words)
            all_wer += wer
            word_len += len(pred_words)

        return {"wer": all_wer, "wer_lens": word_len}
    



@final
@dataclass
class SpeechTextPPLOutput:
    target_tokens: Tensor
    target_mask: Tensor
    logits: Tensor
    pad_idx: Optional[int] = 0
 
    def compute_loss(self, label_smoothing: float = 0.0,) -> Dict[Tensor]:
        lprobs = log_softmax(self.logits, dim=-1, dtype=torch.float32)
        # sum: (), none: (N, S)
        loss = nll_loss(
            lprobs,
            self.target_tokens,
            self.pad_idx,
            label_smoothing=label_smoothing,
            reduction="none",
        )
        num_target_elements = self.target_mask.sum()
        nll_sum = (loss * self.target_mask).sum()
        return nll_sum, num_target_elements
    
    def compute_acc(self, answer_idx):
        num_tokens = self.target_mask.sum(dim=-1) # should be [1,1,1,1] for mmlu
        lprobs = log_softmax(self.logits, dim=-1, dtype=torch.float32)
        target_tokens = self.target_tokens
        target_tokens[~self.target_mask] = 0

        loss = nll_loss(
            lprobs,
            self.target_tokens,
            0,
            reduction="none",
        )
        nll = loss.sum(dim=1)
        nll_token_value, nll_token_idx = (nll / num_tokens).min(dim=-1)
        acc_token = 100.0 * (nll_token_idx == answer_idx)
        return nll_token_value.item(), acc_token
