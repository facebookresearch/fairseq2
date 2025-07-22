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
    TextTokenDecoder,
    TextTokenEncoder,
    TextTokenizer,
    TextTokenizerLoadError,
    text_tokenizer_asset_card_error,
)
from fairseq2.data.text.tokenizers.hg import (
    HuggingFaceTokenDecoder,
    HuggingFaceTokenEncoder,
    HuggingFaceTokenModel,
    load_hg_token_model,
)
from fairseq2.data.text.tokenizers.sentencepiece import (
    load_basic_sentencepiece_tokenizer,
)
from fairseq2.data.text.tokenizers.tiktoken import (
    TiktokenDecoder,
    TiktokenEncoder,
    TiktokenModel,
    load_tiktoken_model,
)
from fairseq2.device import Device

# llama3 chat template with assistant mask support, see https://github.com/huggingface/transformers/issues/28950
LLAMA3_CHAT_TEMPLATE = """{{- bos_token }}
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
class LLaMA3Tokenizer(TextTokenizer):
    """Represents a LLaMA 3 tokenizer."""

    _model: TiktokenModel
    _eos_token: str

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
    def create_decoder(self, *, skip_special_tokens: bool = False) -> TiktokenDecoder:
        return TiktokenDecoder(self._model, skip_special_tokens=skip_special_tokens)

    @property
    @override
    def vocab_info(self) -> VocabularyInfo:
        return self._model.vocab_info


@final
class LLaMA3HuggingFaceTokenizer(TextTokenizer):
    """Represents a Hugging Face version of LLama 3 tokenizer"""

    _model: HuggingFaceTokenModel
    _eos_token: str

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
    ) -> TextTokenEncoder:
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
    ) -> TextTokenEncoder:
        return HuggingFaceTokenEncoder(
            self._model, device=device, pin_memory=pin_memory
        )

    @override
    def create_decoder(self, *, skip_special_tokens: bool = False) -> TextTokenDecoder:
        return HuggingFaceTokenDecoder(
            self._model, skip_special_tokens=skip_special_tokens
        )

    @property
    @override
    def vocab_info(self) -> VocabularyInfo:
        return self._model.vocab_info


LLAMA_TOKENIZER_FAMILY: Final = "llama"


def load_llama_tokenizer(path: Path, card: AssetCard) -> TextTokenizer:

    # first check if this is HuggingFace tokenizer
    try:
        use_hf = card.field("use_hf_tokenizer").as_(bool)
    except AssetCardFieldNotFoundError:
        use_hf = False
    except AssetCardError as ex:
        raise text_tokenizer_asset_card_error(card.name) from ex

    if use_hf:
        try:
            return LLaMA3TokenizerHuggingFace(path)
        except ValueError as ex:
            raise TextTokenizerLoadError(
                card.name, f"The '{card.name}' asset card does not contain a valid text tokenizer configuration of the '{LLAMA_TOKENIZER_FAMILY}' family. See the nested exception for details."  # fmt: skip
            ) from ex
        except RuntimeError as ex:
            raise TextTokenizerLoadError(
                card.name, f"The '{card.name}' text tokenizer cannot be loaded. See the nested exception for details."  # fmt: skip
            ) from ex

    try:
        use_v2 = card.field("use_v2_tokenizer").as_(bool)
    except AssetCardFieldNotFoundError:
        use_v2 = False
    except AssetCardError as ex:
        raise text_tokenizer_asset_card_error(card.name) from ex

    if use_v2:
        try:
            is_dir = path.is_dir()
        except OSError as ex:
            raise TextTokenizerLoadError(
                card.name, f"The path of the '{card.name}' text tokenizer cannot be accessed. See the nested exception for details."  # fmt: skip
            ) from ex

        if is_dir:
            return load_llama3_hg_tokenizer(path, card)

        return load_llama3_tokenizer(path, card)

    return load_basic_sentencepiece_tokenizer(path, card)


def load_llama3_tokenizer(path: Path, card: AssetCard) -> TextTokenizer:
    split_regex = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # fmt: skip

    try:
        use_eot = card.field("use_eot").as_(bool)
    except AssetCardFieldNotFoundError:
        use_eot = False
    except AssetCardError as ex:
        raise text_tokenizer_asset_card_error(card.name) from ex

    # Optionally, the model card can specify a different split_regex (e.g. to support more languages).
    # Extract it from the card or one of its ancestor cards.
    base_card: AssetCard | None = card
    while base_card is not None:
        if base_card.field("split_regex").exists():
            split_regex = card.field("split_regex").as_(str)
            break
        base_card = base_card.base

    eos_token = "<|eot_id|>" if use_eot else "<|end_of_text|>"

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

    try:
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
    except (OSError, RuntimeError) as ex:
        raise TextTokenizerLoadError(
            card.name, f"The '{card.name}' text tokenizer model cannot be loaded. See the nested exception for details."  # fmt: skip
        ) from ex

    return LLaMA3Tokenizer(model, eos_token)


def load_llama3_hg_tokenizer(path: Path, card: AssetCard) -> TextTokenizer:
    try:
        use_eot = card.field("use_eot").as_(bool)
    except AssetCardFieldNotFoundError:
        use_eot = False
    except AssetCardError as ex:
        raise text_tokenizer_asset_card_error(card.name) from ex

    eos_token = "<|eot_id|>" if use_eot else "<|end_of_text|>"

    try:
        model = load_hg_token_model(
            path,
            unk_token=None,
            bos_token="<|begin_of_text|>",
            eos_token=eos_token,
            pad_token="<|finetune_right_pad_id|>",
            boh_token="<|start_header_id|>",
            eoh_token="<|end_header_id|>",
        )
        model.overwrite_chat_template(LLAMA3_CHAT_TEMPLATE)
    except (OSError, RuntimeError) as ex:
        raise TextTokenizerLoadError(
            card.name, f"The '{card.name}' text tokenizer model cannot be loaded. See the nested exception for details."  # fmt: skip
        ) from ex

    return LLaMA3HuggingFaceTokenizer(model, eos_token)
