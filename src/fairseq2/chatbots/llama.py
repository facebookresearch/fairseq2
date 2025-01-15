# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import final

import torch
from torch import Tensor
from typing_extensions import override

from fairseq2.chatbots import (
    AbstractChatbot,
    Chatbot,
    ChatbotHandler,
    ChatDialog,
    ChatMessage,
)
from fairseq2.data.text.tokenizers import TextTokenEncoder, TextTokenizer
from fairseq2.data.text.tokenizers.llama import LLaMA3Tokenizer
from fairseq2.generation import SequenceGenerator
from fairseq2.nn.utils.module import infer_device


@final
class LLaMAChatbot(AbstractChatbot):
    """Represents a LLaMA chatbot."""

    _bos_idx: Tensor
    _eos_idx: Tensor
    _text_encoder: TextTokenEncoder

    def __init__(self, generator: SequenceGenerator, tokenizer: TextTokenizer) -> None:
        """
        :param generator:
            The sequence generator.
        :param tokenizer:
            The text tokenizer.
        """
        super().__init__(generator, tokenizer)

        bos_idx = tokenizer.vocab_info.bos_idx
        eos_idx = tokenizer.vocab_info.eos_idx

        if bos_idx is None or eos_idx is None:
            raise ValueError("`tokenizer` must have BOS and EOS symbols defined.")

        try:
            device = infer_device(generator.model)
        except ValueError as ex:
            raise ValueError(
                "The device of `generator.model` is not valid. See the nested exception for details."
            ) from ex

        self._bos_idx = torch.tensor([bos_idx], device=device)
        self._eos_idx = torch.tensor([eos_idx], device=device)

        self._text_encoder = tokenizer.create_raw_encoder(device=device)

    @override
    def _encode_dialog(self, dialog: ChatDialog, param_name: str) -> Tensor:
        if len(dialog) == 0:
            raise ValueError(
                f"`{param_name}` must have at least one message with the 'user' role."
            )

        if dialog[-1].role != "user":
            raise ValueError(
                f"The last message of `{param_name}` must have the 'user' role."
            )

        # Merge the system message, if any, with the first user message.
        if dialog[0].role == "system":
            content = f"<<SYS>>\n{dialog[0].content}\n<</SYS>>\n\n{dialog[1].content}"

            first_message = ChatMessage(dialog[1].role, content)

            dialog = [first_message] + list(dialog[2:])

        dialog_contents: list[Tensor] = []

        for user, bot in zip(dialog[::2], dialog[1::2]):
            if user.role != "user" or bot.role != "bot":
                raise ValueError(
                    f"The messages of `{param_name}` might optionally start with the 'system' role, and then must alternate between the 'user' and 'bot' roles."
                )

            user_bot_seq = self._text_encoder(
                f"[INST] {user.content.strip()} [/INST] {bot.content.strip()}"
            )

            dialog_contents += [self._bos_idx, user_bot_seq, self._eos_idx]

        user_seq = self._text_encoder(f"[INST] {dialog[-1].content.strip()} [/INST]")

        dialog_contents += [self._bos_idx, user_seq]

        return torch.cat(dialog_contents, dim=0)

    @property
    @override
    def supports_system_prompt(self) -> bool:
        return True


@final
class LLaMA3Chatbot(AbstractChatbot):
    """Represents a LLaMA 3 chatbot."""

    _bos_idx: Tensor
    _eos_idx: Tensor
    _boh_idx: Tensor
    _eoh_idx: Tensor
    _text_encoder: TextTokenEncoder
    _break: Tensor

    def __init__(self, generator: SequenceGenerator, tokenizer: TextTokenizer) -> None:
        """
        :param generator:
            The sequence generator.
        :param tokenizer:
            The text tokenizer.
        """
        super().__init__(generator, tokenizer)

        bos_idx = tokenizer.vocab_info.bos_idx
        eos_idx = tokenizer.vocab_info.eos_idx
        boh_idx = tokenizer.vocab_info.boh_idx
        eoh_idx = tokenizer.vocab_info.eoh_idx

        if bos_idx is None or eos_idx is None or boh_idx is None or eoh_idx is None:
            raise ValueError(
                "`tokenizer` must have BOS, EOS, BOH, EOH symbols defined."
            )

        try:
            device = infer_device(generator.model)
        except ValueError as ex:
            raise ValueError(
                "The device of `generator.model` is not valid. See the nested exception for details."
            ) from ex

        self._bos_idx = torch.tensor([bos_idx], device=device)
        self._eos_idx = torch.tensor([eos_idx], device=device)
        self._boh_idx = torch.tensor([boh_idx], device=device)
        self._eoh_idx = torch.tensor([eoh_idx], device=device)

        self._text_encoder = tokenizer.create_raw_encoder(device=device)

        self._break = self._text_encoder("\n\n")

    @override
    def _encode_dialog(self, dialog: ChatDialog, param_name: str) -> Tensor:
        if len(dialog) == 0:
            raise ValueError(
                f"`{param_name}` must have at least one message with the 'user' role."
            )

        if dialog[-1].role != "user":
            raise ValueError(
                f"The last message of `{param_name}` must have the 'user' role."
            )

        dialog_contents: list[Tensor] = [self._bos_idx]

        def encode_role(role: str) -> None:
            seq = self._text_encoder(role)

            dialog_contents.extend([self._boh_idx, seq, self._eoh_idx, self._break])

        def encode_content(content: str) -> None:
            seq = self._text_encoder(content.strip())

            dialog_contents.extend([seq, self._eos_idx])

        if dialog[0].role == "system":
            encode_role("system")

            encode_content(dialog[0].content)

            dialog = dialog[1:]

        for user, bot in zip(dialog[::2], dialog[1::2]):
            if user.role != "user" or bot.role != "bot":
                raise ValueError(
                    f"The messages of `{param_name}` might optionally start with the 'system' role, and then must alternate between the 'user' and 'bot' roles."
                )

            encode_role("user")

            encode_content(user.content)

            encode_role("assistant")

            encode_content(bot.content)

        encode_role("user")

        encode_content(dialog[-1].content)

        encode_role("assistant")

        return torch.cat(dialog_contents, dim=0)

    @property
    @override
    def supports_system_prompt(self) -> bool:
        return True


@final
class LLaMAChatbotHandler(ChatbotHandler):
    @override
    def create(self, generator: SequenceGenerator, tokenizer: TextTokenizer) -> Chatbot:
        if isinstance(tokenizer, LLaMA3Tokenizer):
            return LLaMA3Chatbot(generator, tokenizer)

        return LLaMAChatbot(generator, tokenizer)
