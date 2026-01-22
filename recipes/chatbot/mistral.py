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

from fairseq2.data.tokenizers import Tokenizer
from fairseq2.device import Device
from fairseq2.utils.tensor import to_tensor

from .chatbot import Dialog, DialogEncoder


@final
class MistralDialogEncoder(DialogEncoder):
    def __init__(self, tokenizer: Tokenizer, device: Device) -> None:
        bos_idx = tokenizer.vocab_info.bos_idx
        eos_idx = tokenizer.vocab_info.eos_idx

        if bos_idx is None or eos_idx is None:
            raise ValueError("`tokenizer` must have BOS and EOS symbols defined.")

        self._bos_idx = to_tensor([bos_idx], device=device)
        self._eos_idx = to_tensor([eos_idx], device=device)

        self._text_encoder = tokenizer.create_raw_encoder(device=device)

    @override
    def encode(self, dialog: Dialog) -> Tensor:
        if len(dialog) == 0:
            raise ValueError(
                "`dialog` must have at least one message with the 'user' role."
            )

        if dialog[-1].role != "user":
            raise ValueError("Last message of `dialog` must have the 'user' role.")

        dialog_contents: list[Tensor] = [self._bos_idx]

        for user, bot in zip(dialog[::2], dialog[1::2]):
            if user.role != "user" or bot.role != "bot":
                raise ValueError(
                    "Messages of `dialog` must alternate between the 'user' and 'bot' roles."
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
