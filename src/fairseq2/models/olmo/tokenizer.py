# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, final

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
from fairseq2.data.tokenizers.sentencepiece import (
    BasicSentencePieceTokenizer,
    load_sentencepiece_model,
)
from fairseq2.data.tokenizers.tiktoken import (
    TiktokenDecoder,
    TiktokenEncoder,
    TiktokenModel,
    load_tiktoken_model,
)
from fairseq2.device import Device
from fairseq2.error import NotSupportedError


@final
class LLaMATiktokenTokenizer(Tokenizer):
    def __init__(self, model: TiktokenModel, eos_token: str) -> None:
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
    ) -> TiktokenEncoder:
        if task is not None:
            raise ValueError(f"`task` must be `None`, but is '{task}' instead.")

        if lang is not None:
            raise ValueError(f"`lang` must be `None`, but is '{lang}' instead.")

        match mode:
            case None | "default":
                prefix_tokens = ["<|begin_of_text|>"]
                suffix_tokens = [self._eos_token]
            case "prompt":
                prefix_tokens = ["<|begin_of_text|>"]
                # In prompt mode, we expect the generator to finish the sequence.
                suffix_tokens = []
            case "prompt_response":
                prefix_tokens = []
                suffix_tokens = [self._eos_token]
            case "as_is":
                prefix_tokens = []
                suffix_tokens = []
            case _:
                raise NotSupportedError(
                    f"`mode` must be a supported mode, but is {mode} instead. Supported modes are default, prompt, prompt_response, as_is."
                )

        return TiktokenEncoder(
            self._model,
            prefix_tokens=prefix_tokens,
            suffix_tokens=suffix_tokens,
            device=device,
            pin_memory=pin_memory,
        )

    @override
    def create_raw_encoder(
        self, *, device: Device | None = None, pin_memory: bool = False
    ) -> TiktokenEncoder:
        return TiktokenEncoder(self._model, device=device, pin_memory=pin_memory)

    @override
    def create_decoder(self, *, skip_special_tokens: bool = False) -> TiktokenDecoder:
        return TiktokenDecoder(self._model, skip_special_tokens=skip_special_tokens)

    @property
    @override
    def vocab_info(self) -> VocabularyInfo:
        return self._model.vocab_info


# Chat template with assistant mask support, see
# https://github.com/huggingface/transformers/issues/28950
LLAMA3_HG_CHAT_TEMPLATE: Final = """{{- bos_token }}
{%- for message in messages %}
    {%- if message.role == 'user' or message.role == 'system' %}
        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' + message['content'] | trim + '<|eot_id|>' }}
    {%- else %}
        {%- generation %}
        {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' + message['content'] | trim + '<|eot_id|>' }}
        {%- endgeneration %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}
{%- endif %}"""


@final
class LLaMAHuggingFaceTokenizer(Tokenizer):
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

        match mode:
            case None | "default":
                prefix_tokens = ["<|begin_of_text|>"]
                suffix_tokens = [self._eos_token]
            case "prompt":
                prefix_tokens = ["<|begin_of_text|>"]
                # In prompt mode, we expect the generator to finish the sequence.
                suffix_tokens = []
            case "prompt_response":
                prefix_tokens = []
                suffix_tokens = [self._eos_token]
            case "as_is":
                prefix_tokens = []
                suffix_tokens = []
            case _:
                raise NotSupportedError(
                    f"`mode` must be a supported mode, but is {mode} instead. Supported modes are default, prompt, prompt_response, as_is."
                )

        return HuggingFaceTokenEncoder(
            self._model,
            prefix_tokens=prefix_tokens,
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


@dataclass
class LLaMATokenizerConfig:
    impl: Literal["sp", "tiktoken", "hg"] = "sp"
    use_eot: bool = False
    split_regex: str | None = None


def load_llama_tokenizer(path: Path, config: LLaMATokenizerConfig) -> Tokenizer:
    match config.impl:
        case "sp":
            return _load_llama_sp_tokenizer(path, config)
        case "tiktoken":
            return _load_llama_tt_tokenizer(path, config)
        case "hg":
            return _load_llama_hg_tokenizer(path, config)
        case _:
            raise ValueError(
                f"`config.impl` must be 'sp', 'tiktoken', or 'hg', but is '{config.impl}' instead."
            )


def _load_llama_sp_tokenizer(path: Path, config: LLaMATokenizerConfig) -> Tokenizer:
    if config.use_eot:
        raise NotSupportedError(
            "LLaMA SentencePiece tokenizer does not support `config.use_eot`."
        )

    if config.split_regex is not None:
        raise NotSupportedError(
            "LLaMA SentencePiece tokenizer does not support `config.split_regex`."
        )

    model = load_sentencepiece_model(path)

    return BasicSentencePieceTokenizer(model)


def _load_llama_tt_tokenizer(path: Path, config: LLaMATokenizerConfig) -> Tokenizer:
    if config.split_regex is None:
        split_regex = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # fmt: skip
    else:
        split_regex = config.split_regex

    eos_token = "<|eot_id|>" if config.use_eot else "<|end_of_text|>"

    special_tokens = [
        "<|begin_of_text|>",
        "<|end_of_text|>",
        "<|reserved_special_token_0|>",
        "<|reserved_special_token_1|>",
        "<|finetune_right_pad_id|>",
        "<|step_id|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eom_id|>",  # end-of-message
        "<|eot_id|>",  # end-of-turn
        "<|python_tag|>",
    ]

    num_reserved_special_tokens = 256

    for i in range(num_reserved_special_tokens - len(special_tokens)):
        special_tokens.append(f"<|reserved_special_token_{2 + i}|>")

    model = load_tiktoken_model(
        path,
        split_regex,
        unk_token=None,
        bos_token="<|begin_of_text|>",
        eos_token=eos_token,
        pad_token="<|finetune_right_pad_id|>",
        boh_token="<|start_header_id|>",
        eoh_token="<|end_header_id|>",
        special_tokens=special_tokens,
    )

    return LLaMATiktokenTokenizer(model, eos_token)


def _load_llama_hg_tokenizer(path: Path, config: LLaMATokenizerConfig) -> Tokenizer:
    if config.split_regex is not None:
        raise NotSupportedError(
            "LLaMA Hugging Face tokenizer does not support `config.split_regex`."
        )

    eos_token = "<|eot_id|>" if config.use_eot else "<|end_of_text|>"

    model = load_hg_token_model(
        path,
        unk_token=None,
        bos_token="<|begin_of_text|>",
        eos_token=eos_token,
        pad_token="<|finetune_right_pad_id|>",
        boh_token="<|start_header_id|>",
        eoh_token="<|end_header_id|>",
    )

    model.overwrite_chat_template(LLAMA3_HG_CHAT_TEMPLATE)

    return LLaMAHuggingFaceTokenizer(model, eos_token)
