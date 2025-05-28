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
    Chatbot,
    ChatbotHandler,
    ChatDialog,
    ChatDialogEncoder,
    StandardChatbot,
)
from fairseq2.data.text.tokenizers import TextTokenEncoder, TextTokenizer
from fairseq2.generation import SequenceGenerator
from fairseq2.models.clm import CausalLM
from fairseq2.models.mistral import MISTRAL_MODEL_FAMILY
from fairseq2.nn.utils.module import infer_device
from fairseq2.utils.tensor import to_tensor


@final
class MistralDialogEncoder(ChatDialogEncoder):
    _bos_idx: Tensor
    _eos_idx: Tensor
    _text_encoder: TextTokenEncoder

    def __init__(self, model: CausalLM, tokenizer: TextTokenizer) -> None:
        bos_idx = tokenizer.vocab_info.bos_idx
        eos_idx = tokenizer.vocab_info.eos_idx

        if bos_idx is None or eos_idx is None:
            raise ValueError("`tokenizer` must have BOS and EOS symbols defined.")

        try:
            device = infer_device(model)
        except ValueError as ex:
            raise ValueError(
                "The device of `generator.model` is not valid. See the nested exception for details."
            ) from ex

        self._bos_idx = to_tensor([bos_idx], device=device)
        self._eos_idx = to_tensor([eos_idx], device=device)

        self._text_encoder = tokenizer.create_raw_encoder(device=device)

    @override
    def encode(self, dialog: ChatDialog) -> Tensor:
        if len(dialog) == 0:
            raise ValueError(
                "`dialog` must have at least one message with the 'user' role."
            )

        if dialog[-1].role != "user":
            raise ValueError("The last message of `dialog` must have the 'user' role.")

        dialog_contents: list[Tensor] = [self._bos_idx]

        for user, bot in zip(dialog[::2], dialog[1::2]):
            if user.role != "user" or bot.role != "bot":
                raise ValueError(
                    "The messages of `dialog` must alternate between the 'user' and 'bot' roles."
                )

            user_bot_seq = self._text_encoder(
                f"[INST] {user.content.strip()} [/INST] {bot.content.strip()}"
            )

            dialog_contents += [user_bot_seq, self._eos_idx]

        user_seq = self._text_encoder(f"[INST] {dialog[-1].content.strip()} [/INST]")

        dialog_contents.append(user_seq)

        return torch.cat(dialog_contents, dim=0)


@final
class MistralChatbotHandler(ChatbotHandler):
    @override
    def create(self, generator: SequenceGenerator, tokenizer: TextTokenizer) -> Chatbot:
        dialog_encoder = MistralDialogEncoder(generator.model, tokenizer)

        text_decoder = tokenizer.create_decoder(skip_special_tokens=True)

        return StandardChatbot(
            generator, dialog_encoder, text_decoder, supports_system_prompt=False
        )

    @property
    @override
    def family(self) -> str:
        return MISTRAL_MODEL_FAMILY
