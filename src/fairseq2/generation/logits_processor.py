# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from fairseq2.data.text.text_tokenizer import TextTokenEncoder
from fairseq2.data.typing import StringLike
from fairseq2.typing import Device


class LogitsProcessor(ABC):
    """Abstracte base class for updating scores in place"""

    @abstractmethod
    def __call__(self, seqs: Tensor, lprobs: Tensor) -> None:
        """Update next-step log probabilities inplace based on given token sequence.

        :param seqs:
            The sequence of tokens generated in current beam search step.
            :math:`(N,B,S)`, where :math:`N` is the batch size, :math:`B` is
            the number of beams, and :math:`S` is the size of the sequence.
        :param lprobs:
            The next-step log probability of each vocabulary entry. *Shape:*
            :math:`(N,B,V)`, where :math:`N` is the batch size, :math:`B` is
            the number of beams, and :math:`V` is the size of the vocabulary.

        :returns:
            None
        """


class BannedSequenceLogitsProcessor(LogitsProcessor):
    """Processor used to penalize scores of multiple banned sequences of words."""

    banned_tokens: Tensor
    """Vector of shape (nb_banned_sequences, 1) containing last token of each sequence to ban."""

    banned_prefix: Tensor
    """Matrix of shape (nb_banned_sequences, max_banned_tokens_len - 1) padded with 0s on the left."""

    banned_prefix_mask: Tensor
    """mask of 0s and 1s based on each banned token sequence and max prefix len."""

    max_prefix_len: int
    """length of biggest banned sequence - 1."""

    pad_idx: int
    """Padding index used for encoding banned sequences."""

    device: Device
    """device used for all inner tensors."""

    def __init__(self, banned_seqs: List[Tensor], pad_idx: int, device: Device) -> None:
        """
        :param banned_seqs:
            list of token sequences to ban.
        :param pad_idx:
            padding index used for encoding banned sequences.
        :param device:
            device
        """
        if len(banned_seqs) == 0:
            raise ValueError("`banned_seqs` should contain at least one element.")
        if any([t.ndim != 1 for t in banned_seqs]):
            raise ValueError(
                "`banned_seqs` should contain only one dimensional tensors."
            )

        self.pad_idx = pad_idx
        self.device = device

        self.max_prefix_len = max([len(t) - 1 for t in banned_seqs])
        self.banned_prefix = self._create_pad_tensor(
            size=(len(banned_seqs), self.max_prefix_len)
        )
        self.banned_tokens = torch.empty(
            size=(len(banned_seqs), 1), dtype=torch.int64, device=self.device
        )
        for i, seq in enumerate(banned_seqs):
            if (len(seq)) > 1:
                self.banned_prefix[i, -len(seq) + 1 :] = seq[:-1]
            self.banned_tokens[i] = seq[-1]

        self.banned_prefix_mask = torch.where(
            self.banned_prefix == self.pad_idx, 0, 1
        ).to(device=self.device)

    def __call__(self, seqs: Tensor, lprobs: Tensor) -> None:
        """Apply score penalty of banend tokens inplace"""
        seqs = self._pad_left_short_sequence(seqs)

        if self.max_prefix_len == 0:
            lprobs[:, :, self.banned_tokens] = -torch.inf
        else:
            prefix_diff = (
                seqs[:, :, -self.max_prefix_len :].unsqueeze(2)
                * self.banned_prefix_mask
                - self.banned_prefix
            )
            batch_idx, beam_idx, match_idx = (prefix_diff.sum(dim=-1) == 0).nonzero(
                as_tuple=True
            )
            if len(batch_idx) > 0:
                lprobs[batch_idx, beam_idx, self.banned_tokens[match_idx]] = -torch.inf

    def _pad_left_short_sequence(self, tokens: Tensor) -> Tensor:
        batch_size, beam_size, seq_len = tokens.shape
        if seq_len < self.max_prefix_len:
            tmp = self._create_pad_tensor(
                size=(batch_size, beam_size, self.max_prefix_len)
            )
            tmp[:, :, -seq_len:] = tokens
            tokens = tmp

        return tokens

    def _create_pad_tensor(self, size: Tuple[int, ...]) -> Tensor:
        return torch.full(
            size=size,
            fill_value=self.pad_idx,
            dtype=torch.int64,
            device=self.device,
        )

    # This is not the best place but the whole file needs a refactoring
    # We need target decoder to create this tensor
    @staticmethod
    def compute_banned_words_seqs(
        banned_strings: Sequence[StringLike],
        token_encoder: TextTokenEncoder,
    ) -> List[Tensor]:
        """Compute sequences of tokens to ban from encoder and banned strings

        :params banned_strings:
            The list of strings to ban in sequence generation.
        :params token_encoder:
            Encoder to use for tokenizing input strings.

        :returns:
            List of token sequences to ban.
        """
        if not banned_strings:
            return []

        control_tokens = BannedSequenceLogitsProcessor._concat_optional_tensors(
            [token_encoder.prefix_indices, token_encoder.suffix_indices]
        )

        def encode(s: StringLike) -> torch.Tensor:
            seq = token_encoder(s)
            if control_tokens is None:
                return seq

            mask = torch.isin(seq, control_tokens, invert=True)
            return seq[mask]

        return [encode(x) for x in banned_strings]

    @staticmethod
    def _concat_optional_tensors(tensors: List[Optional[Tensor]]) -> Optional[Tensor]:
        not_none_tensors = [t for t in tensors if t is not None]

        result: Optional[Tensor] = None
        if len(not_none_tensors) > 0:
            result = torch.cat(not_none_tensors).unique()

        return result
