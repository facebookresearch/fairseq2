# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Sequence

from torch import Tensor

from fairseq2.data.typing import StringLike
from fairseq2.data.vocabulary_info import VocabularyInfo
from fairseq2.typing import Device


class TextTokenizer(ABC):
    """Represents a tokenizer to encode and decode text."""

    vocab_info: VocabularyInfo

    def __init__(self, vocab_info: VocabularyInfo) -> None:
        """
        :param vocab_info:
            The vocabulary information associated with the tokenizer.
        """
        self.vocab_info = vocab_info

    @abstractmethod
    def create_encoder(
        self,
        *,
        task: Optional[str] = None,
        lang: Optional[str] = None,
        mode: Optional[str] = None,
        device: Optional[Device] = None,
        pin_memory: bool = False,
    ) -> TextTokenEncoder:
        """Create a token encoder.

        The valid arguments for the ``task``, ``lang``, and ``mode`` parameters
        are implementation specific. Refer to concrete ``TextTokenizer``
        subclasses for more information.

        :param task:
            The task for which to generate token indices. Typically, multi-task
            jobs use ``task`` to distinguish between different tasks such as
            'translation' or 'transcription'.
        :param lang:
            The language of generated token indices. Typically, multilingual
            translation tasks use ``lang`` to distinguish between different
            languages such as 'en-US' or 'de-DE'.
        :param mode:
            The mode in which to generate token indices. Typically, translation
            tasks use ``mode`` to distinguish between different modes such as
            'source' or 'target'.
        :param device:
            The device on which to construct tensors.
        :param pin_memory:
            If ``True``, uses pinned memory while constructing tensors.
        """

    @abstractmethod
    def create_raw_encoder(
        self, *, device: Optional[Device] = None, pin_memory: bool = False
    ) -> TextTokenEncoder:
        """Create a raw token encoder with no control symbols.

        :param device:
            The device on which to construct tensors.
        :param pin_memory:
            If ``True``, uses pinned memory while constructing tensors.
        """

    @abstractmethod
    def create_decoder(self) -> TextTokenDecoder:
        """Create a token decoder."""


class TextTokenEncoder(ABC):
    """Encodes text into tokens or token indices."""

    @abstractmethod
    def __call__(self, text: StringLike) -> Tensor:
        """
        :param text:
            The text to encode.
        """

    @abstractmethod
    def encode_as_tokens(self, text: StringLike) -> List[StringLike]:
        """
        :param text:
            The text to encode.
        """

    @property
    @abstractmethod
    def prefix_indices(self) -> Optional[Tensor]:
        """Get the indices of the prefix tokens. *Shape:* :math:`(S)`, where
        :math:`S` is the number of indices."""

    @property
    @abstractmethod
    def suffix_indices(self) -> Optional[Tensor]:
        """Get the indices of the suffix tokens. *Shape:* :math:`(S)`, where
        :math:`S` is the number of indices."""


class TextTokenDecoder(ABC):
    """Decodes text from tokens or token indices."""

    @abstractmethod
    def __call__(self, token_indices: Tensor) -> StringLike:
        """
        :param token_indices:
            The token indices to decode from.
        """

    @abstractmethod
    def decode_from_tokens(self, tokens: Sequence[StringLike]) -> StringLike:
        """
        :param tokens:
            The tokens to decode from.
        """
