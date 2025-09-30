# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias, final

from torch import Tensor
from typing_extensions import override

from fairseq2.data.tokenizers import TokenDecoder
from fairseq2.error import InternalError
from fairseq2.generation import SequenceGenerator, SequenceGeneratorOutput
from fairseq2.nn import BatchLayout


@dataclass
class Message:
    """Represents a message exchanged between a user and a bot."""

    role: Literal["system", "user", "bot"]
    """The party that sent this message in the dialog."""

    content: str
    """The message content."""


Dialog: TypeAlias = Sequence[Message]


class Chatbot(ABC):
    """Represents a chatbot."""

    @abstractmethod
    def response(self, dialog: Dialog) -> tuple[Message, SequenceGeneratorOutput]:
        """
        :param dialog: The dialog that the bot should respond to.

        :returns:
            - The response message of the bot.
            - The output of the underlying sequence generator.
        """

    @property
    @abstractmethod
    def supports_system_prompt(self) -> bool:
        """Whether the chatbot supports an initial system prompt."""


class DialogEncoder(ABC):
    @abstractmethod
    def encode(self, dialog: Dialog) -> Tensor: ...

    @property
    @abstractmethod
    def supports_system_prompt(self) -> bool: ...


@final
class StandardChatbot(Chatbot):
    """Provides a standard implementation of :class:`Chatbot`."""

    def __init__(
        self,
        generator: SequenceGenerator,
        dialog_encoder: DialogEncoder,
        text_decoder: TokenDecoder,
    ) -> None:
        self._generator = generator
        self._dialog_encoder = dialog_encoder
        self._text_decoder = text_decoder

    @override
    def response(self, dialog: Dialog) -> tuple[Message, SequenceGeneratorOutput]:
        dialog_seq = self._dialog_encoder.encode(dialog)

        # (S) -> (1, S)
        dialog_seqs = dialog_seq.unsqueeze(0)

        dialog_seqs_layout = BatchLayout.of(dialog_seqs)

        generator_output = self._generator(dialog_seqs, dialog_seqs_layout)

        hypotheses = generator_output.hypotheses[0]
        if len(hypotheses) == 0:
            raise InternalError("Sequence generator returned no hypothesis.")

        response = Message(role="bot", content=self._text_decoder(hypotheses[0].seq))

        return response, generator_output

    @property
    @override
    def supports_system_prompt(self) -> bool:
        return self._dialog_encoder.supports_system_prompt
