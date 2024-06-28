# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, final

import torch
from torch import Tensor

from fairseq2.data.text import TextTokenEncoder, TextTokenizer
from fairseq2.generation import AbstractChatbot, ChatDialog, SequenceGenerator
from fairseq2.models.chatbot import create_chatbot
from fairseq2.models.mistral.factory import MISTRAL_FAMILY
from fairseq2.nn.utils.module import infer_device
from fairseq2.typing import override


@final
class MistralChatbot(AbstractChatbot):
    """Represents a Mistral chatbot."""

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

        dialog_contents: List[Tensor] = [self._bos_idx]

        for user, bot in zip(dialog[::2], dialog[1::2]):
            if user.role != "user" or bot.role != "bot":
                raise ValueError(
                    f"The messages of `{param_name}` must alternate between the roles 'user' and 'bot'."
                )

            user_bot_seq = self._text_encoder(
                f"[INST] {user.content.strip()} [/INST] {bot.content.strip()}"
            )

            dialog_contents += [user_bot_seq, self._eos_idx]

        user_seq = self._text_encoder(f"[INST] {dialog[-1].content.strip()} [/INST]")

        dialog_contents.append(user_seq)

        return torch.cat(dialog_contents, dim=0)

    @property
    @override
    def supports_system_prompt(self) -> bool:
        return False


create_chatbot.register(MISTRAL_FAMILY, MistralChatbot)
