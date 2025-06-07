# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import final

from typing_extensions import override

from fairseq2.data.tokenizers import (
    TokenDecoder,
    TokenEncoder,
    Tokenizer,
    TokenizerLoadError,
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
    SentencePieceModel,
)
from fairseq2.data.tokenizers.tiktoken import (
    TiktokenDecoder,
    TiktokenEncoder,
    TiktokenModel,
    load_tiktoken_model,
)
from fairseq2.device import Device
from fairseq2.error import InfraError
from fairseq2.file_system import FileSystem
from fairseq2.runtime.dependency import DependencyResolver


@final
class LLaMA3Tokenizer(Tokenizer):
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
class LLaMA3HuggingFaceTokenizer(Tokenizer):
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
    use_v2_tokenizer: bool = False
    use_eot: bool = False


def _load_llama_tokenizer(
    resolver: DependencyResolver, path: Path, name: str, config: LLaMATokenizerConfig
) -> Tokenizer:
    if config.use_v2_tokenizer:
        file_system = resolver.resolve(FileSystem)

        try:
            is_dir = file_system.is_dir(path)
        except OSError as ex:
            raise InfraError(
                f"A system error has occurred while accessing the path of the '{name}' tokenizer. See the nested exception for details."  # fmt: skip
            ) from ex

        if is_dir:
            return _load_llama3_hg_tokenizer(path, name, config)

        return _load_llama3_tokenizer(path, name, config)

    try:
        model = SentencePieceModel(path)
    except OSError as ex:
        raise InfraError(
            f"A system error has occurred while reading the '{path}' tokenizer model. See the nested exception for details."
        ) from ex
    except RuntimeError as ex:
        raise TokenizerLoadError(
            name, f"The '{path}' tokenizer model cannot be loaded. See the nested exception for details."  # fmt: skip
        ) from ex

    return BasicSentencePieceTokenizer(model)


def _load_llama3_tokenizer(
    path: Path, name: str, config: LLaMATokenizerConfig
) -> Tokenizer:
    split_regex = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # fmt: skip

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
    except OSError as ex:
        raise InfraError(
            f"A system error has occurred while reading the '{path}' tokenizer model. See the nested exception for details."
        ) from ex
    except RuntimeError as ex:
        raise TokenizerLoadError(
            name, f"The '{path}' tokenizer model cannot be loaded. See the nested exception for details."  # fmt: skip
        ) from ex

    return LLaMA3Tokenizer(model, eos_token)


def _load_llama3_hg_tokenizer(
    path: Path, name: str, config: LLaMATokenizerConfig
) -> Tokenizer:
    eos_token = "<|eot_id|>" if config.use_eot else "<|end_of_text|>"

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
    except OSError as ex:
        raise InfraError(
            f"A system error has occurred while reading the '{path}' tokenizer model. See the nested exception for details."
        ) from ex
    except RuntimeError as ex:
        raise TokenizerLoadError(
            name, f"The '{path}' tokenizer model cannot be loaded. See the nested exception for details."  # fmt: skip
        ) from ex

    return LLaMA3HuggingFaceTokenizer(model, eos_token)
