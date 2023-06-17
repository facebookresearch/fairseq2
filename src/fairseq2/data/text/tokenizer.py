# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch
from torch import Tensor

from fairseq2.data.typing import StringLike
from fairseq2.typing import DataType, Device


class Tokenizer(ABC):
    """Represents a tokenizer to encode and decode sentences."""

    vocab_info: "VocabularyInfo"

    def __init__(self, vocab_info: "VocabularyInfo") -> None:
        """
        :param vocab_info:
            The vocabulary information associated with the tokenizer.
        """
        self.vocab_info = vocab_info

    @abstractmethod
    def create_encoder(
        self,
        task: Optional[str] = None,
        lang: Optional[str] = None,
        mode: Optional[str] = None,
        batch_size: Optional[int] = None,
        device: Optional[Device] = None,
        pin_memory: bool = False,
        dtype: DataType = torch.int64,
        disable_parallelism: bool = False,
    ) -> "TokenEncoder":
        """Create a token encoder.

        The valid arguments for the ``task``, ``lang``, and ``mode`` parameters
        are implementation specific. Refer to concrete ``Tokenizer`` subclasses
        for more information.

        :param task:
            An optional implementation-specific task such as 'translation' or
            'transcription' for which to generate token indices.
        :param lang:
            An optional implementation-specific identifier indicating the
            language of generated token indices. Typically used by multilingual
            tokenizers for distinguishing between different source and target
            languages.
        :param mode:
            An optional implementation-specific mode in which to generate token
            indices. Typically used by translation tasks to indicate whether the
            encoding is done for source or target sentences.
        :param batch_size:
            If the number of sentences to encode is less than ``batch_size``,
            the output will be padded.
        :param device:
            The device on which to initialize token indices.
        :param pin_memory:
            If ``True``, uses pinned memory before copying token indices to the
            target device. (only supported by CUDA devices)
        :param dtype:
            The integral data type of generated token indices.
        :param disabled_parallelism:
            If ``True``, disables parallelism and uses the calling thread only.
        """

    @abstractmethod
    def create_decoder(self) -> "TokenDecoder":
        """Create a token decoder."""
        pass


@dataclass(frozen=True)
class VocabularyInfo:
    size: int
    """The size of the vocabulary."""

    unk_idx: int
    """The index of the symbol that represents an unknown word."""

    bos_idx: int
    """The index of the symbol that represents the beginning of a sentence."""

    eos_idx: int
    """The index of the symbol that represents the end of a sentence."""

    pad_idx: int
    """The index of the symbol that is used to pad a sentence."""


class TokenEncoder(ABC):
    """Encodes sentences into token indices."""

    @abstractmethod
    def __call__(self, sentences: Union[StringLike, Sequence[StringLike]]) -> Tensor:
        """
        :param sentences:
            The sentences to encode.
        """


class TokenDecoder(ABC):
    """Decodes sentences from token indices."""

    @abstractmethod
    def __call__(self, token_indices: Tensor) -> Sequence[StringLike]:
        """
        :param token_indices:
            The token indices to decode from.
        """
