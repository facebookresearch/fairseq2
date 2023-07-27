# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Sequence

import torch
from torch import Tensor

from fairseq2.data import VocabularyInfo


class UnitTokenizer:
    """Represents a tokenizer to encode and decode UnitY speech units."""

    num_units: int
    langs: Dict[str, int]

    def __init__(self, num_units: int, langs: Sequence[str]) -> None:
        """
        :param num_units:
            The number of speech units.
        :param langs:
            The list of supported languages.
        """
        self.num_units = num_units

        self.langs = {lang: idx for idx, lang in enumerate(langs)}

        # For legacy reasons, we have to repeat the language symbols twice along
        # with a placeholder `<mask>` token.
        vocab_size = num_units + (2 * (len(langs) + 1)) + 4

        # We use fairseq's control symbol order.
        self.vocabulary_info = VocabularyInfo(
            size=vocab_size, bos_idx=0, pad_idx=1, eos_idx=2, unk_idx=3
        )

    def lang_to_index(self, lang: str) -> int:
        """Return the symbol index of the specified language."""
        return self.num_units + len(self.langs) + self.langs[lang] + 5

    def create_encoder(self, lang: str) -> "UnitTokenEncoder":
        """Create a token encoder.

        :param lang:
            The language of generated token indices.
        """
        return UnitTokenEncoder(self, lang)

    def create_decoder(self) -> "UnitTokenDecoder":
        """Create a token decoder."""
        return UnitTokenDecoder(self)


class UnitTokenEncoder:
    """Encodes speech units into token indices."""

    tokenizer: UnitTokenizer
    eos_idx: int
    unk_idx: int
    lang_idx: int

    def __init__(self, tokenizer: UnitTokenizer, lang: str) -> None:
        """
        :param tokenizer:
            The unit tokenizer to use.
        :param lang:
            The language of generated token indices.
        """
        if not lang in tokenizer.langs:
            langs = ", ".join(tokenizer.langs.keys())

            raise ValueError(
                f"`lang` must be one of the supported languages, but is '{lang}' instead. Supported languages: {langs}"
            )

        self.tokenizer = tokenizer

        assert tokenizer.vocabulary_info.eos_idx is not None
        assert tokenizer.vocabulary_info.unk_idx is not None

        self.eos_idx = tokenizer.vocabulary_info.eos_idx
        self.unk_idx = tokenizer.vocabulary_info.unk_idx

        self.lang_idx = tokenizer.lang_to_index(lang)

    def __call__(self, units: Tensor) -> Tensor:
        """Encode ``units`` to token indices.

        :param units:
            The speech units to encode. *Shape:* :math:`(N,S)`, where :math:`N`
            is the batch size and :math:`S` is the sequence length.

        :returns:
            The token indices corresponding to ``units``. *Shape:*
            :math:`(N,S_{tok})` ,where :math:`N` is the batch size and
            :math`S_{tok}` is the sequence length of the token indices.
        """
        batch_size = units.size(0)

        # We always start sequences with EOS, followed by the language token.
        prefix_seq = torch.tensor(
            [self.eos_idx, self.lang_idx], device=units.device, dtype=units.dtype
        )

        token_indices = torch.cat(
            [prefix_seq.expand(batch_size, -1), units.detach()], dim=1
        )

        # Ensure that non-symbol indices larger than `num_units` are replaced
        # with UNK.
        seqs = token_indices[:, 2:]

        seqs[seqs >= self.tokenizer.num_units] = self.unk_idx

        return token_indices


class UnitTokenDecoder:
    """Decodes speech units from token indices."""

    eos_idx: int
    pad_idx: int

    def __init__(self, tokenizer: UnitTokenizer) -> None:
        """
        :param tokenizer:
            The unit tokenizer to use.
        """
        assert tokenizer.vocabulary_info.eos_idx is not None
        assert tokenizer.vocabulary_info.pad_idx is not None

        self.eos_idx = tokenizer.vocabulary_info.eos_idx
        self.pad_idx = tokenizer.vocabulary_info.pad_idx

    def __call__(self, token_indices: Tensor) -> Tensor:
        """Decode ``token_indices`` to speech units.

        :param token_indices:
            The token indices to decode. *Shape:* :math:`(N,S)`, where :math:`N`
            is the batch size and :math:`S` is the sequence length.

        :returns:
            The speech units corresponding to ``token_indices``. *Shape:*
            :math:`(N,S_{unt})`, where :math:`N` is the batch size and
            :math`S_{unt}` is the sequence length of the speech units.
        """
        if token_indices.size(1) == 0:
            return token_indices

        # Remove the prefix EOS symbol. The language symbol is still expected to
        # be part of the decoded output.
        units = token_indices[:, 1:].clone().detach()

        # Also, replace EOS with PAD at sequence ends.
        units[units == self.eos_idx] = self.pad_idx

        return units
