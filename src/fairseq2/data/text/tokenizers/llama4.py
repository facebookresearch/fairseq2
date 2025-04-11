# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from pathlib import Path
from typing import Final, final

from typing_extensions import override

from fairseq2.assets import AssetCard, AssetCardError, AssetCardFieldNotFoundError
from fairseq2.data import VocabularyInfo
from fairseq2.data.text.tokenizers import (
    TextTokenizer,
    TextTokenizerLoadError,
    text_tokenizer_asset_card_error,
)
from fairseq2.data.text.tokenizers.tiktoken import (
    TiktokenDecoder,
    TiktokenEncoder,
    TiktokenModel,
)
from fairseq2.typing import Device



def get_reserved_special_tokens(name, count, start_index=0):
    return [f"<|{name}_reserved_special_token_{i}|>" for i in range(start_index, start_index + count)]


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
    LLAMA4_TEXT_POST_TRAIN_SPECIAL_TOKENS + LLAMA4_VISION_SPECIAL_TOKENS + LLAMA4_REASONING_SPECIAL_TOKENS
)

BASIC_SPECIAL_TOKENS = [
    "<|begin_of_text|>",
    "<|end_of_text|>",
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
]


@final
class LLaMA4Tokenizer(TextTokenizer):
    """Represents a LLaMA 4 tokenizer."""
    
    O200K_PATTERN = r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"""  # fmt: skip
    
    num_reserved_special_tokens = 2048

    _SPLIT_REGEX: Final = O200K_PATTERN

    _model: TiktokenModel
    _eos_token: str
    special_tokens: dict[str, int]

    def __init__(self, path: Path, custom_eos: str | None = "<|eot|>") -> None:
        """
        :param path:
            The path to the tiktoken BPE file.
        :param custom_eos:
            If not ``None``, replaces the original EOS token.
        """
        self._eos_token = custom_eos or "<|end_of_text|>"
        
        special_tokens = BASIC_SPECIAL_TOKENS + LLAMA4_SPECIAL_TOKENS
        assert len(set(special_tokens)) == len(special_tokens)
        assert len(special_tokens) <= self.num_reserved_special_tokens

        reserved_tokens = [
            f"<|reserved_special_token_{i}|>" for i in range(self.num_reserved_special_tokens - len(special_tokens))
        ]
        special_tokens = special_tokens + reserved_tokens

        self._model = TiktokenModel(
            path,
            split_regex=self._SPLIT_REGEX,
            unk_token=None,
            bos_token="<|begin_of_text|>",
            eos_token=self._eos_token,
            pad_token="<|finetune_right_pad|>",
            boh_token="<|header_start|>",
            eoh_token="<|header_end|>",
            special_tokens=special_tokens,
        )
        self.special_tokens = self._model.special_tokens

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
                raise ValueError(
                    f"`mode` must be one of the following values, but is '{mode}' instead: default, prompt, prompt_response, as_is"
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
    def create_decoder(self) -> TiktokenDecoder:
        return TiktokenDecoder(self._model)

    @property
    @override
    def vocab_info(self) -> VocabularyInfo:
        return self._model.vocab_info


LLAMA4_TOKENIZER_FAMILY: Final = "llama4"


def load_llama4_tokenizer(path: Path, card: AssetCard) -> TextTokenizer:
    try:
        return LLaMA4Tokenizer(path)
    except ValueError as ex:
        raise TextTokenizerLoadError(
            card.name, f"The '{card.name}' asset card does not contain a valid text tokenizer configuration of the '{LLAMA4_TOKENIZER_FAMILY}' family. See the nested exception for details."  # fmt: skip
        ) from ex
    except RuntimeError as ex:
        raise TextTokenizerLoadError(
            card.name, f"The '{card.name}' text tokenizer cannot be loaded. See the nested exception for details."  # fmt: skip
        ) from ex
