# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Final

import torch

from fairseq2.models.llama import load_llama_model, load_llama_tokenizer
from tests.common import device

PROMPT: Final = "I believe"

TARGET: Final = "I believe that the best way to learn is by doing. I have been teaching for 10 years and I have taught students of all ages and levels."

MAX_LEN: Final = 30


def test_load_llama_7b() -> None:
    model = load_llama_model("llama_7b", device=device, progress=False)

    tokenizer = load_llama_tokenizer("llama", progress=False)

    model.eval()

    tokens = tokenizer.create_encoder()(PROMPT)

    prefix = torch.LongTensor([tokenizer.vocab_info.bos_idx], device=device)

    tokens = torch.cat((prefix, tokens), dim=0).unsqueeze(0)

    for _ in range(MAX_LEN):
        decoder_out, decoder_padding_mask = model.decode(tokens, seq_lens=None)

        logits = model.project(decoder_out, decoder_padding_mask).logits

        next_token = torch.argmax(logits[:, -1:None, :], dim=2)

        tokens = torch.cat([tokens, next_token], dim=1)

    generated = str(tokenizer.create_decoder()(tokens)[0])

    assert generated == TARGET
