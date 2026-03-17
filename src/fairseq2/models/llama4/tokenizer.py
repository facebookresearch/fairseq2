# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fairseq2.data.tokenizers import Tokenizer
from fairseq2.data.tokenizers.tiktoken import load_tiktoken_model
from fairseq2.models.llama.tokenizer import LLaMATiktokenTokenizer


def get_reserved_special_tokens(
    name: str, count: int, start_index: int = 0
) -> list[str]:
    return [
        f"<|{name}_reserved_special_token_{i}|>"
        for i in range(start_index, start_index + count)
    ]


# 200005, ..., 200079
LLAMA4_TEXT_POST_TRAIN_SPECIAL_TOKENS = [
    "<|header_start|>",
    "<|header_end|>",
    "<|eom|>",
    "<|eot|>",
    "<|step|>",
    "<|text_post_train_reserved_special_token_0|>",
    "<|text_post_train_reserved_special_token_1|>",
    "<|text_post_train_reserved_special_token_2|>",
    "<|text_post_train_reserved_special_token_3|>",
    "<|text_post_train_reserved_special_token_4|>",
    "<|text_post_train_reserved_special_token_5|>",
    "<|text_post_train_reserved_special_token_6|>",
    "<|text_post_train_reserved_special_token_7|>",
    "<|finetune_right_pad|>",
] + get_reserved_special_tokens(
    "text_post_train", 61, 8
)  # <|text_post_train_reserved_special_token_6|>, ..., <|text_post_train_reserved_special_token_66|>

# 200080, ..., 201133
LLAMA4_VISION_SPECIAL_TOKENS = [
    "<|image_start|>",
    "<|image_end|>",
    "<|vision_reserved_special_token_0|>",
    "<|vision_reserved_special_token_1|>",
    "<|tile_x_separator|>",
    "<|tile_y_separator|>",
    "<|vision_reserved_special_token_2|>",
    "<|vision_reserved_special_token_3|>",
    "<|vision_reserved_special_token_4|>",
    "<|vision_reserved_special_token_5|>",
    "<|image|>",
    "<|vision_reserved_special_token_6|>",
    "<|patch|>",
] + get_reserved_special_tokens(
    "vision", 1041, 7
)  # <|vision_reserved_special_token_7|>, ..., <|vision_reserved_special_token_1047|>

# 201134, ..., 201143
LLAMA4_REASONING_SPECIAL_TOKENS = [
    "<|reasoning_reserved_special_token_0|>",
    "<|reasoning_reserved_special_token_1|>",
    "<|reasoning_reserved_special_token_2|>",
    "<|reasoning_reserved_special_token_3|>",
    "<|reasoning_reserved_special_token_4|>",
    "<|reasoning_reserved_special_token_5|>",
    "<|reasoning_reserved_special_token_6|>",
    "<|reasoning_reserved_special_token_7|>",
    "<|reasoning_thinking_start|>",
    "<|reasoning_thinking_end|>",
]

LLAMA4_SPECIAL_TOKENS = (
    LLAMA4_TEXT_POST_TRAIN_SPECIAL_TOKENS
    + LLAMA4_VISION_SPECIAL_TOKENS
    + LLAMA4_REASONING_SPECIAL_TOKENS
)

BASIC_SPECIAL_TOKENS = [
    "<|begin_of_text|>",
    "<|end_of_text|>",
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
]


@dataclass
class Llama4TokenizerConfig:
    use_eot: bool = True
    split_regex: str | None = None


def load_llama4_tokenizer(path: Path, config: Llama4TokenizerConfig) -> Tokenizer:
    if config.split_regex is None:
        O200K_PATTERN = r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"""  # fmt: skip
        split_regex = O200K_PATTERN
    else:
        split_regex = config.split_regex

    eos_token = "<|eot|>" if config.use_eot else "<|end_of_text|>"

    num_reserved_special_tokens = 2048

    special_tokens = BASIC_SPECIAL_TOKENS + LLAMA4_SPECIAL_TOKENS

    if len(set(special_tokens)) != len(special_tokens):
        raise ValueError(
            "There are unexpected duplicates in the tokenizer's special tokens."
        )

    if len(special_tokens) > num_reserved_special_tokens:
        raise ValueError(
            f"The number of special tokens ({len(special_tokens)}) exceeds the number of reserved special tokens ({num_reserved_special_tokens})."
        )

    reserved_tokens = [
        f"<|reserved_special_token_{i}|>"
        for i in range(num_reserved_special_tokens - len(special_tokens))
    ]
    special_tokens = special_tokens + reserved_tokens

    model = load_tiktoken_model(
        path,
        split_regex,
        unk_token=None,
        bos_token="<|begin_of_text|>",
        eos_token=eos_token,
        pad_token="<|finetune_right_pad|>",
        boh_token="<|header_start|>",
        eoh_token="<|header_end|>",
        special_tokens=special_tokens,
    )

    return LLaMATiktokenTokenizer(model, eos_token)
