# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import final

from torch import Tensor
from typing_extensions import override

from fairseq2.data._vocabulary_info import VocabularyInfo
from fairseq2.typing import Device


class TextTokenizer(ABC):
    """Represents a tokenizer to encode and decode text."""

    @abstractmethod
    def create_encoder(
        self,
        *,
        task: str | None = None,
        lang: str | None = None,
        mode: str | None = None,
        device: Device | None = None,
        pin_memory: bool = False,
    ) -> TextTokenEncoder:
        """Constructs a token encoder.

        The valid arguments for the ``task``, ``lang``, and ``mode`` parameters
        are implementation specific. Refer to concrete ``TextTokenizer``
        subclasses for more information.

        :param task:
            The task for which to generate token indices. Typically, ``task`` is
            used to distinguish between different tasks such as 'translation' or
            'transcription'.
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
        self, *, device: Device | None = None, pin_memory: bool = False
    ) -> TextTokenEncoder:
        """Constructs a raw token encoder with no control symbols.

        :param device:
            The device on which to construct tensors.
        :param pin_memory:
            If ``True``, uses pinned memory while constructing tensors.
        """

    @abstractmethod
    def create_decoder(self) -> TextTokenDecoder:
        """Constructs a token decoder."""

    @property
    @abstractmethod
    def vocab_info(self) -> VocabularyInfo:
        """The vocabulary information associated with the tokenizer."""


class AbstractTextTokenizer(TextTokenizer):
    """Provides a skeletal implementation of :class:`TextTokenizer`."""

    _vocab_info: VocabularyInfo

    def __init__(self, vocab_info: VocabularyInfo) -> None:
        """
        :param vocab_info:
            The vocabulary information associated with the tokenizer.
        """
        self._vocab_info = vocab_info

    @final
    @property
    @override
    def vocab_info(self) -> VocabularyInfo:
        """The vocabulary information associated with the tokenizer."""
        return self._vocab_info


class TextTokenEncoder(ABC):
    """Encodes text into tokens or token indices."""

    @abstractmethod
    def __call__(self, text: str) -> Tensor:
        """
        :param text:
            The text to encode.
        """

    @abstractmethod
    def encode_as_tokens(self, text: str) -> list[str]:
        """
        :param text:
            The text to encode.
        """

    @property
    @abstractmethod
    def prefix_indices(self) -> Tensor | None:
        """Get the indices of the prefix tokens. *Shape:* :math:`(S)`, where
        :math:`S` is the number of indices."""

    @property
    @abstractmethod
    def suffix_indices(self) -> Tensor | None:
        """Get the indices of the suffix tokens. *Shape:* :math:`(S)`, where
        :math:`S` is the number of indices."""


class TextTokenDecoder(ABC):
    """Decodes text from tokens or token indices."""

    @abstractmethod
    def __call__(self, token_indices: Tensor) -> str:
        """
        :param token_indices:
            The token indices to decode from.
        """

    @abstractmethod
    def decode_from_tokens(self, tokens: Sequence[str]) -> str:
        """
        :param tokens:
            The tokens to decode from.
        """
