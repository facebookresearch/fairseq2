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
    ChatDialog,
    ChatMessage,
    SequenceGenerator,
    create_chatbot,
)
from fairseq2.nn.utils.module import infer_device
from fairseq2.typing import override


@final
class LLaMAChatbot(AbstractChatbot):
    """Represents a LLaMA chatbot."""

    _bos_idx: Tensor
    _eos_idx: Tensor
    _text_encoder: TextTokenEncoder

    def __init__(
        self,
        generator: SequenceGenerator,
        tokenizer: TextTokenizer,
        *,
        stdout: bool = False,
    ) -> None:
        """
        :param generator:
            The sequence generator.
        :param tokenizer:
            The text tokenizer.
        :param stdout:
            If ``True``, prints generated messages to stdout in real-time.
        """
        super().__init__(generator, tokenizer, stdout=stdout)

        assert tokenizer.vocab_info.bos_idx is not None
        assert tokenizer.vocab_info.eos_idx is not None

        device = infer_device(generator.model, param_name="generator.model")

        self._bos_idx = torch.tensor([tokenizer.vocab_info.bos_idx], device=device)
        self._eos_idx = torch.tensor([tokenizer.vocab_info.eos_idx], device=device)

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


create_chatbot.register("llama", LLaMAChatbot)
