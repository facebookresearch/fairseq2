# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, final

from typing_extensions import override

from fairseq2.data.tokenizers import (
    TokenDecoder,
    TokenEncoder,
    Tokenizer,
    VocabularyInfo,
)
from fairseq2.data.tokenizers.hg import (
    HuggingFaceTokenDecoder,
    HuggingFaceTokenEncoder,
    HuggingFaceTokenModel,
    load_hg_token_model,
)
from fairseq2.device import Device
from fairseq2.error import NotSupportedError

# Chat template with assistant mask support.

## this article shows the hg implementation on
# https://huggingface.co/allenai/OLMo-2-1124-7B-Instruct/blob/main/tokenizer_config.json

OLMO_HG_CHAT_TEMPLATE: Final = """{{ bos_token }}
{% for message in messages %}
{% if message['role'] == 'system' %}
{{ '<|system|>\n' + message['content'] + '\n' }}
{% elif message['role'] == 'user' %}
{{ '<|user|>\n' + message['content'] + '\n' }}
{% elif message['role'] == 'assistant' %}{% if not loop.last %}
{{ '<|assistant|>\n'  + message['content'] + eos_token + '\n' }}
{% else %}
{{ '<|assistant|>\n'  + message['content'] + eos_token }}{% endif %}{% endif %}
{% if loop.last and add_generation_prompt %}
{{ '<|assistant|>\n' }}{% endif %}
{% endfor %}
"""

# chat template example
"""
# <|endoftext|><|user|>
How are you doing?
<|assistant|>
I'm just a computer program, so I don't have feelings, but I'm functioning as expected. How can I assist you today?<|endoftext|>
"""


@final
class OlmoTokenizer(Tokenizer):
    def __init__(self, model: HuggingFaceTokenModel, eos_token: str) -> None:
        self._model = model
        self._eos_token = eos_token

    @override
    def create_encoder(
        self,
        *,
        task: str | None = None,
        lang: str | None = None,
        mode: str | None = None,
        device: Device | None = None,
        pin_memory: bool = False,
    ) -> TokenEncoder:
        if task is not None:
            raise ValueError(f"`task` must be `None`, but is '{task}' instead.")

        if lang is not None:
            raise ValueError(f"`lang` must be `None`, but is '{lang}' instead.")

        # OLMo2 doesn't use prefix tokens (unlike LLaMA which uses <|begin_of_text|>)
        # The BOS token is added via the chat template when needed
        match mode:
            case None | "default":
                suffix_tokens = [self._eos_token]
            case "prompt":
                # In prompt mode, we expect the generator to finish the sequence.
                suffix_tokens = []
            case "prompt_response":
                suffix_tokens = [self._eos_token]
            case "as_is":
                suffix_tokens = []
            case _:
                raise NotSupportedError(
                    f"`mode` must be a supported mode, but is {mode} instead. Supported modes are default, prompt, prompt_response, as_is."
                )

        return HuggingFaceTokenEncoder(
            self._model,
            prefix_tokens=[],
            suffix_tokens=suffix_tokens,
            device=device,
            pin_memory=pin_memory,
        )

    @override
    def create_raw_encoder(
        self, *, device: Device | None = None, pin_memory: bool = False
    ) -> TokenEncoder:
        return HuggingFaceTokenEncoder(
            self._model, device=device, pin_memory=pin_memory
        )

    @override
    def create_decoder(self, *, skip_special_tokens: bool = False) -> TokenDecoder:
        return HuggingFaceTokenDecoder(
            self._model, skip_special_tokens=skip_special_tokens
        )

    @property
    @override
    def vocab_info(self) -> VocabularyInfo:
        return self._model.vocab_info


@dataclass(kw_only=True)
class OlmoTokenizerConfig:
    use_im_end: bool = False


def load_olmo_tokenizer(path: Path, config: OlmoTokenizerConfig) -> Tokenizer:
    # Use <|im_end|> as eos_token if configured, otherwise use <|endoftext|>
    # Similar to Qwen's tokenizer configuration
    eos_token = "<|im_end|>" if config.use_im_end else "<|endoftext|>"

    model = load_hg_token_model(
        path,
        unk_token="<|endoftext|>",
        bos_token="<|endoftext|>",
        eos_token=eos_token,
        pad_token="<|pad|>",
        boh_token=None,
        eoh_token=None,
    )

    model.overwrite_chat_template(OLMO_HG_CHAT_TEMPLATE)

    return OlmoTokenizer(model, eos_token)
