# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from torch import Tensor

from fairseq2.data.tokenizers.vocab_info import VocabularyInfo
from fairseq2.device import Device


class Tokenizer(ABC):
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
    ) -> TokenEncoder:
        """Constructs a token encoder.

        The valid arguments for the ``task``, ``lang``, and ``mode`` parameters
        are implementation specific. Refer to concrete ``Tokenizer``
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
    ) -> TokenEncoder:
        """Constructs a raw token encoder with no control symbols.

        :param device: The device on which to construct tensors.
        :param pin_memory: If ``True``, uses pinned memory for tensors.
        """

    @abstractmethod
    def create_decoder(self, *, skip_special_tokens: bool = False) -> TokenDecoder:
        """Constructs a token decoder."""

    @property
    @abstractmethod
    def vocab_info(self) -> VocabularyInfo:
        """The vocabulary information associated with the tokenizer."""


class TokenEncoder(ABC):
    """Encodes text into tokens or token indices."""

    @abstractmethod
    def __call__(self, text: str) -> Tensor:
        """
        :param text: The text to encode.
        """

    @abstractmethod
    def encode_as_tokens(self, text: str) -> list[str]:
        """
        :param text: The text to encode.
        """

    @property
    @abstractmethod
    def prefix_indices(self) -> Tensor | None:
        """
        Gets the indices of the prefix tokens. *Shape:* :math:`(S)`, where
        :math:`S` is the number of indices.
        """

    @property
    @abstractmethod
    def suffix_indices(self) -> Tensor | None:
        """
        Gets the indices of the suffix tokens. *Shape:* :math:`(S)`, where
        :math:`S` is the number of indices.
        """


class TokenDecoder(ABC):
    """Decodes text from tokens or token indices."""

    @abstractmethod
    def __call__(self, token_indices: Tensor) -> str: ...

    @abstractmethod
    def decode_from_tokens(self, tokens: Sequence[str]) -> str: ...
