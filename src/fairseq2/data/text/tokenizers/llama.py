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
from fairseq2.data.text.tokenizers.sentencepiece import BasicSentencePieceTokenizer
from fairseq2.data.text.tokenizers.tiktoken import (
    TiktokenDecoder,
    TiktokenEncoder,
    TiktokenModel,
)
from fairseq2.data.text.tokenizers.huggingface_tokenizer import (
    HuggingfaceTokenizerEncoder,
    HuggingfaceTokenizerDecoder,
)
from fairseq2.typing import Device
from transformers import AutoTokenizer


@final
class LLaMA3TokenizerHuggingFace(TextTokenizer):
    """Represents a HuggingFace version of LLama 3 tokenizer"""

    _tokenizer: AutoTokenizer
    _bos_token: str
    _eos_token: str

    def __init__(self, path: Path) -> None:

        self._tokenizer = AutoTokenizer.from_pretrained(path)

        self._eos_token = self._tokenizer.special_tokens_map["eos_token"]
        self._bos_token = self._tokenizer.special_tokens_map["bos_token"]

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
                prefix_tokens = [self._bos_token]
                suffix_tokens = [self._eos_token]
            case "prompt":
                prefix_tokens = [self._bos_token]
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

        return HuggingfaceTokenizerEncoder(
            self._tokenizer,
            prefix_tokens=prefix_tokens,
            suffix_tokens=suffix_tokens,
            device=device,
            pin_memory=pin_memory,
        )

    @override
    def create_raw_encoder(
        self, *, device: Device | None = None, pin_memory: bool = False
    ) -> TiktokenEncoder:
        return HuggingfaceTokenizerEncoder(
            self._tokenizer, device=device, pin_memory=pin_memory
        )

    @override
    def create_decoder(self) -> TiktokenDecoder:
        return HuggingfaceTokenizerDecoder(self._model)

    @property
    @override
    def vocab_info(self) -> VocabularyInfo:
        bos_idx = self._tokenizer.convert_tokens_to_ids(self._bos_token)
        eos_idx = self._tokenizer.convert_tokens_to_ids(self._eos_token)
        vocab_info = VocabularyInfo(
            size=len(self._tokenizer),
            bos_idx=bos_idx,
            eos_idx=eos_idx,
            unk_idx=None,
            pad_idx=None,
        )
        return vocab_info


@final
class LLaMA3Tokenizer(TextTokenizer):
    """Represents a LLaMA 3 tokenizer."""

    _SPLIT_REGEX: Final = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # fmt: skip

    _model: TiktokenModel
    _eos_token: str

    def __init__(self, path: Path, use_eot: bool = False) -> None:
        """
        :param path:
            The path to the tiktoken BPE file.
        :param use_eot:
            If ``True``, uses EOT (end-of-turn) token in-place of EOS token.
        """
        self._eos_token = "<|eot_id|>" if use_eot else "<|end_of_text|>"

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

        self._model = TiktokenModel(
            path,
            split_regex=self._SPLIT_REGEX,
            unk_token=None,
            bos_token="<|begin_of_text|>",
            eos_token=self._eos_token,
            pad_token="<|finetune_right_pad_id|>",
            boh_token="<|start_header_id|>",
            eoh_token="<|end_header_id|>",
            special_tokens=special_tokens,
        )

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
            use_eot = card.field("use_eot").as_(bool)
        except AssetCardFieldNotFoundError:
            use_eot = False
        except AssetCardError as ex:
            raise text_tokenizer_asset_card_error(card.name) from ex

        try:
            return LLaMA3Tokenizer(path, use_eot=use_eot)
        except ValueError as ex:
            raise TextTokenizerLoadError(
                card.name, f"The '{card.name}' asset card does not contain a valid text tokenizer configuration of the '{LLAMA_TOKENIZER_FAMILY}' family. See the nested exception for details."  # fmt: skip
            ) from ex
        except RuntimeError as ex:
            raise TextTokenizerLoadError(
                card.name, f"The '{card.name}' text tokenizer cannot be loaded. See the nested exception for details."  # fmt: skip
            ) from ex
    else:
        try:
            return BasicSentencePieceTokenizer(path)
        except RuntimeError as ex:
            raise TextTokenizerLoadError(
                card.name, f"The '{card.name}' text tokenizer cannot be loaded. See the nested exception for details."  # fmt: skip
            ) from ex
