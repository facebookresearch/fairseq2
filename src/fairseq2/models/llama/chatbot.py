# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, final

import torch
from torch import Tensor

from fairseq2.data.text import TextTokenEncoder, TextTokenizer
from fairseq2.generation import (
    AbstractChatbot,
    Chatbot,
    ChatDialog,
    ChatMessage,
    SequenceGenerator,
)
from fairseq2.models.chatbot import create_chatbot
from fairseq2.models.llama.factory import LLAMA_FAMILY
from fairseq2.models.llama.tokenizer import LLaMA3Tokenizer
from fairseq2.nn.utils.module import infer_device
from fairseq2.typing import override


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
            raise RuntimeError(
                "One or more required control symbols requierd for the chatbot are not found in the tokenizer. Please make sure that you are using the right tokenizer."
            )

        device = infer_device(generator.model, name="generator.model")

        self._bos_idx = torch.tensor([bos_idx], device=device)
        self._eos_idx = torch.tensor([eos_idx], device=device)

        self._text_encoder = tokenizer.create_raw_encoder(device=device)

    @override
    def _encode_dialog(self, dialog: ChatDialog, param_name: str) -> Tensor:
        if len(dialog) == 0:
            raise ValueError(
                f"`{param_name}` must have at least one message with the role 'user'."
            )

        if dialog[-1].role != "user":
            raise ValueError(
                f"The last message of `{param_name}` must have the role 'user'."
            )

        # Merge the system message, if any, with the first user message.
        if dialog[0].role == "system":
            content = f"<<SYS>>\n{dialog[0].content}\n<</SYS>>\n\n{dialog[1].content}"

            first_message = ChatMessage(dialog[1].role, content)

            dialog = [first_message] + list(dialog[2:])

        dialog_contents: List[Tensor] = []

        for user, bot in zip(dialog[::2], dialog[1::2]):
            if user.role != "user" or bot.role != "bot":
                raise ValueError(
                    f"The messages of `{param_name}` might optionally start with the role 'system', and then must alternate between the roles 'user' and 'bot'."
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
    _boh_idx: Tensor
    _eoh_idx: Tensor
    _eot_idx: Tensor
    _text_encoder: TextTokenEncoder
    _break: Tensor

    def __init__(
        self, generator: SequenceGenerator, tokenizer: LLaMA3Tokenizer
    ) -> None:
        """
        :param generator:
            The sequence generator.
        :param tokenizer:
            The text tokenizer.
        """
        super().__init__(generator, tokenizer)

        device = infer_device(generator.model, name="generator.model")

        try:
            bos_idx = tokenizer.encoding.encode_single_token("<|begin_of_text|>")
            boh_idx = tokenizer.encoding.encode_single_token("<|start_header_id|>")
            eoh_idx = tokenizer.encoding.encode_single_token("<|end_header_id|>")
            eot_idx = tokenizer.encoding.encode_single_token("<|eot_id|>")
        except KeyError:
            raise RuntimeError(
                "One or more special symbols required for the chatbot are not found in the tokenizer. Please file a bug report."
            )

        self._bos_idx = torch.tensor([bos_idx], device=device)
        self._boh_idx = torch.tensor([boh_idx], device=device)
        self._eoh_idx = torch.tensor([eoh_idx], device=device)
        self._eot_idx = torch.tensor([eot_idx], device=device)

        self._text_encoder = tokenizer.create_raw_encoder(device=device)

        self._break = self._text_encoder("\n\n")

    @override
    def _encode_dialog(self, dialog: ChatDialog, param_name: str) -> Tensor:
        if len(dialog) == 0:
            raise ValueError(
                f"`{param_name}` must have at least one message with the role 'user'."
            )

        if dialog[-1].role != "user":
            raise ValueError(
                f"The last message of `{param_name}` must have the role 'user'."
            )

        dialog_contents: List[Tensor] = [self._bos_idx]

        def encode_role(role: str) -> None:
            seq = self._text_encoder(role)

            dialog_contents.extend([self._boh_idx, seq, self._eoh_idx, self._break])

        def encode_content(content: str) -> None:
            seq = self._text_encoder(content.strip())

            dialog_contents.extend([seq, self._eot_idx])

        if dialog[0].role == "system":
            encode_role("system")

            encode_content(dialog[0].content)

            dialog = dialog[1:]

        for user, bot in zip(dialog[::2], dialog[1::2]):
            if user.role != "user" or bot.role != "bot":
                raise ValueError(
                    f"The messages of `{param_name}` might optionally start with the role 'system', and then must alternate between the roles 'user' and 'bot'."
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


def create_llama_chatbot(
    generator: SequenceGenerator, tokenizer: TextTokenizer
) -> Chatbot:
    """Create the appropriate LLaMA chatbot based on ``tokenizer``."""
    if isinstance(tokenizer, LLaMA3Tokenizer):
        return LLaMA3Chatbot(generator, tokenizer)

    return LLaMAChatbot(generator, tokenizer)


create_chatbot.register(LLAMA_FAMILY, create_llama_chatbot)
