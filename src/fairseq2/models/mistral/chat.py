# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, final

import torch
from torch import Tensor

from fairseq2.data.text import TextTokenEncoder
from fairseq2.generation import Chatbot, ChatDialog, SequenceGenerator
from fairseq2.models.mistral.tokenizer import MistralTokenizer
from fairseq2.nn.utils.module import infer_device
from fairseq2.typing import finaloverride


@final
class MistralChatbot(Chatbot):
    """Represents a Mistral chatbot."""

    bos_idx: Tensor
    eos_idx: Tensor
    text_encoder: TextTokenEncoder

    def __init__(
        self, generator: SequenceGenerator, tokenizer: MistralTokenizer
    ) -> None:
        """
        :param generator:
            The sequence generator.
        :param tokenizer:
            The text tokenizer.
        """
        super().__init__(generator, tokenizer)

        assert tokenizer.vocab_info.bos_idx is not None
        assert tokenizer.vocab_info.eos_idx is not None

        device = infer_device(generator.model)

        self.bos_idx = torch.tensor([tokenizer.vocab_info.bos_idx], device=device)
        self.eos_idx = torch.tensor([tokenizer.vocab_info.eos_idx], device=device)

        self.text_encoder = tokenizer.create_raw_encoder(device=device)

    @finaloverride
    def _encode_dialog(self, dialog: ChatDialog, param_name: str) -> Tensor:
        if len(dialog) == 0:
            raise ValueError(
                f"`{param_name}` must have at least one message with the role 'user'."
            )

        if dialog[-1].role != "user":
            raise ValueError(
                f"The last message of `{param_name}` must have the role 'user'."
            )

        dialog_contents: List[Tensor] = [self.bos_idx]

        for user, bot in zip(dialog[::2], dialog[1::2]):
            if user.role != "user" or bot.role != "bot":
                raise ValueError(
                    f"The messages of `{param_name}` must alternate between the roles 'user' and 'bot'."
                )

            user_bot_seq = self.text_encoder(
                f"[INST] {user.content.strip()} [/INST] {bot.content.strip()}"
            )

            dialog_contents += [user_bot_seq, self.eos_idx]

        user_seq = self.text_encoder(f"[INST] {dialog[-1].content.strip()} [/INST]")

        dialog_contents.append(user_seq)

        return torch.cat(dialog_contents, dim=0)
