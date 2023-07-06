# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence

from torch import Tensor

from fairseq2.data.typing import StringLike
from fairseq2.typing import Device


class Tokenizer(ABC):
    """Represents a tokenizer to encode and decode sentences."""

    vocabulary_info: "VocabularyInfo"

    def __init__(self, vocabulary_info: "VocabularyInfo") -> None:
        """
        :param vocabulary_info:
            The vocabulary information associated with the tokenizer.
        """
        self.vocabulary_info = vocabulary_info

    @abstractmethod
    def create_encoder(
        self,
        task: Optional[str] = None,
        lang: Optional[str] = None,
        mode: Optional[str] = None,
        device: Optional[Device] = None,
        pin_memory: bool = False,
    ) -> "TokenEncoder":
        """Create a token encoder.

        The valid arguments for the ``task``, ``lang``, and ``mode`` parameters
        are implementation specific. Refer to concrete ``Tokenizer`` subclasses
        for more information.

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
    def create_decoder(self) -> "TokenDecoder":
        """Create a token decoder."""


@dataclass(frozen=True)
class VocabularyInfo:
    size: int
    """The size of the vocabulary."""

    unk_idx: Optional[int]
    """The index of the symbol that represents an unknown word."""

    bos_idx: Optional[int]
    """The index of the symbol that represents the beginning of a sentence."""

    eos_idx: Optional[int]
    """The index of the symbol that represents the end of a sentence."""

    pad_idx: Optional[int]
    """The index of the symbol that is used to pad a sentence."""


class TokenEncoder(ABC):
    """Encodes sentences into token indices."""

    @abstractmethod
    def __call__(self, sentence: StringLike) -> Tensor:
        """
        :param sentence:
            The sentence to encode.
        """


class TokenDecoder(ABC):
    """Decodes sentences from token indices."""

    @abstractmethod
    def __call__(self, token_indices: Tensor) -> Sequence[StringLike]:
        """
        :param token_indices:
            The token indices to decode from.
        """
