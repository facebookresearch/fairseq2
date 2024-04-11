# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, ContextManager, List, Literal, Optional, Sequence, Tuple, final

from torch import Tensor
from typing_extensions import TypeAlias

from fairseq2.data.text import TextTokenDecoder, TextTokenizer
from fairseq2.generation.generator import SequenceGenerator, SequenceGeneratorOutput
from fairseq2.generation.utils import _StdOutPrintHook
from fairseq2.nn.padding import PaddingMask, pad_seqs
from fairseq2.typing import override


@final
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
    def __call__(
        self, dialog: ChatDialog, *, stdout: bool = False
    ) -> Tuple[ChatMessage, SequenceGeneratorOutput]:
        """
        :param dialog:
            The chat dialog that the bot should respond to.
        :param stdout:
            If ``True``, prints the generated message in real-time to stdout.

        :returns:
            - The response message of the bot.
            - The output of the underlying sequence generator.
        """

    @abstractmethod
    def batch_response(
        self, dialogs: Sequence[ChatDialog]
    ) -> Tuple[List[ChatMessage], SequenceGeneratorOutput]:
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


class AbstractChatbot(Chatbot):
    """Provides a skeletal implementation of :class:`Chatbot`."""

    _generator: SequenceGenerator
    _text_decoder: TextTokenDecoder

    def __init__(self, generator: SequenceGenerator, tokenizer: TextTokenizer) -> None:
        """
        :param generator:
            The sequence generator.
        :param tokenizer:
            The text tokenizer.
        """
        self._generator = generator

        self._text_decoder = tokenizer.create_decoder()

    @final
    @override
    def __call__(
        self, dialog: ChatDialog, *, stdout: bool = False
    ) -> Tuple[ChatMessage, SequenceGeneratorOutput]:
        dialog_seq = self._encode_dialog(dialog, "dialog")

        cm: ContextManager[Any]

        if stdout:
            hook = _StdOutPrintHook(self._text_decoder)

            cm = self._generator.register_step_hook(hook)
        else:
            cm = nullcontext()

        with cm:
            responses, generator_output = self.__do_response(
                dialog_seq.unsqueeze(0), dialog_padding_mask=None
            )

        return responses[0], generator_output

    @final
    @override
    def batch_response(
        self, dialogs: Sequence[ChatDialog]
    ) -> Tuple[List[ChatMessage], SequenceGeneratorOutput]:
        """
        :param dialogs:
            The chat dialogs that the bot should respond to.

        :returns:
            - The response messages of the bot.
            - The output of the underlying sequence generator.
        """
        dialog_seq_list = [
            self._encode_dialog(d, f"dialogs[{i}]") for i, d in enumerate(dialogs)
        ]

        dialog_seqs, dialog_padding_mask = pad_seqs(dialog_seq_list)

        return self.__do_response(dialog_seqs, dialog_padding_mask)

    def __do_response(
        self, dialog_seqs: Tensor, dialog_padding_mask: Optional[PaddingMask]
    ) -> Tuple[List[ChatMessage], SequenceGeneratorOutput]:
        generator_output = self._generator(dialog_seqs, dialog_padding_mask)

        responses: List[ChatMessage] = []

        for idx, hypotheses in enumerate(generator_output.hypotheses):
            if len(hypotheses) == 0:
                raise RuntimeError(
                    f"The sequence generator returned no hypothesis at index {idx}. Please file a bug report."
                )

            response = ChatMessage(
                role="bot", content=self._text_decoder(hypotheses[0].seq)
            )

            responses.append(response)

        return responses, generator_output

    @abstractmethod
    def _encode_dialog(self, dialog: ChatDialog, param_name: str) -> Tensor:
        """Encodes ``dialog`` to pass to the underlying sequence generator.

        :param dialog:
            The dialog to encode.
        :param param_name:
            The parameter name to use in case of an argument error.
        """
