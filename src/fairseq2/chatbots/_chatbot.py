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

from fairseq2.data.text.tokenizers import TextTokenDecoder
from fairseq2.error import ContractError
from fairseq2.generation import SequenceGenerator, SequenceGeneratorOutput
from fairseq2.nn import BatchLayout
from fairseq2.nn.utils.padding import pad_seqs


@dataclass
class ChatMessage:
    """Represents a chat message exchanged between a user and a bot."""

    role: Literal["system", "user", "bot"]
    """The party that sent this message in the dialog."""

    content: str
    """The message content."""


ChatDialog: TypeAlias = Sequence[ChatMessage]


class Chatbot(ABC):
    """Represents a chatbot."""

    @abstractmethod
    def response(
        self, dialog: ChatDialog
    ) -> tuple[ChatMessage, SequenceGeneratorOutput]:
        """
        :param dialog:
            The chat dialog that the bot should respond to.

        :returns:
            - The response message of the bot.
            - The output of the underlying sequence generator.
        """

    @abstractmethod
    def batch_response(
        self, dialogs: Sequence[ChatDialog]
    ) -> tuple[list[ChatMessage], SequenceGeneratorOutput]:
        """
        :param dialogs:
            The chat dialogs that the bot should respond to.

        :returns:
            - The response messages of the bot.
            - The output of the underlying sequence generator.
        """

    @property
    @abstractmethod
    def supports_system_prompt(self) -> bool:
        """Whether the chatbot supports an initial system prompt."""


class ChatDialogEncoder(ABC):
    @abstractmethod
    def encode(self, dialog: ChatDialog) -> Tensor: ...


@final
class StandardChatbot(Chatbot):
    """Provides a standard implementation of :class:`Chatbot`."""

    _generator: SequenceGenerator
    _dialog_encoder: ChatDialogEncoder
    _text_decoder: TextTokenDecoder
    _supports_system_prompt: bool

    def __init__(
        self,
        generator: SequenceGenerator,
        dialog_encoder: ChatDialogEncoder,
        text_decoder: TextTokenDecoder,
        supports_system_prompt: bool,
    ) -> None:
        self._generator = generator
        self._dialog_encoder = dialog_encoder
        self._text_decoder = text_decoder
        self._supports_system_prompt = supports_system_prompt

    @override
    def response(
        self, dialog: ChatDialog
    ) -> tuple[ChatMessage, SequenceGeneratorOutput]:
        dialog_seq = self._dialog_encoder.encode(dialog)

        # (S) -> (1, S)
        dialog_seqs = dialog_seq.unsqueeze(0)

        dialog_seqs_layout = BatchLayout.of(dialog_seqs)

        responses, generator_output = self._generate_response(
            dialog_seqs, dialog_seqs_layout
        )

        return responses[0], generator_output

    @override
    def batch_response(
        self, dialogs: Sequence[ChatDialog]
    ) -> tuple[list[ChatMessage], SequenceGeneratorOutput]:
        """
        :param dialogs:
            The chat dialogs that the bot should respond to.

        :returns:
            - The response messages of the bot.
            - The output of the underlying sequence generator.
        """
        dialog_seqs = []

        for idx, dialog in enumerate(dialogs):
            try:
                dialog_seq = self._dialog_encoder.encode(dialog)
            except ValueError as ex:
                raise ValueError(
                    "`dialogs[{idx}]` is not valid. See the nested exception for details."
                ) from ex

            dialog_seqs.append(dialog_seq)

        dialog_seqs_pt, dialog_seqs_layout = pad_seqs(dialog_seqs)

        return self._generate_response(dialog_seqs_pt, dialog_seqs_layout)

    def _generate_response(
        self, dialog_seqs: Tensor, dialog_seqs_layout: BatchLayout
    ) -> tuple[list[ChatMessage], SequenceGeneratorOutput]:
        generator_output = self._generator(dialog_seqs, dialog_seqs_layout)

        responses: list[ChatMessage] = []

        for idx, hypotheses in enumerate(generator_output.hypotheses):
            if len(hypotheses) == 0:
                raise ContractError(
                    f"The sequence generator returned no hypothesis at index {idx}."
                )

            response = ChatMessage(
                role="bot", content=self._text_decoder(hypotheses[0].seq)
            )

            responses.append(response)

        return responses, generator_output

    @property
    @override
    def supports_system_prompt(self) -> bool:
        return self._supports_system_prompt
