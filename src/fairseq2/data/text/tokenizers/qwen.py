# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from pathlib import Path
from typing import Final, final

from typing_extensions import override

from fairseq2.assets import AssetCard
from fairseq2.data import VocabularyInfo
from fairseq2.data.text.tokenizers import (
    TextTokenizer,
    TextTokenizerLoadError,
)
from fairseq2.data.text.tokenizers.tiktoken import (
    TiktokenDecoder,
    TiktokenEncoder,
)
from fairseq2.data.text.tokenizers.huggingface_tokenizer import (
    HuggingfaceTokenizerEncoder,
    HuggingfaceTokenizerDecoder,
)
from fairseq2.typing import Device

try:
    from transformers import AutoTokenizer
except ImportError:
    raise RuntimeError(
        "transformers library is required to use HF tokenizers. Install it via `pip install transformers`."
    )


@final
class QwenTokenizerHuggingFace(TextTokenizer):
    """Represents a HuggingFace version of LLama 3 tokenizer"""

    _tokenizer: AutoTokenizer
    _bos_token: str
    _eos_token: str

    def __init__(self, path: Path) -> None:

        self._tokenizer = AutoTokenizer.from_pretrained(path)

        self._eos_token = self._tokenizer.special_tokens_map["eos_token"]
        self._bos_token = None

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
                prefix_tokens = []
                suffix_tokens = [self._eos_token]
            case "prompt":
                prefix_tokens = []
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
        return HuggingfaceTokenizerDecoder(self._tokenizer)

    @property
    @override
    def vocab_info(self) -> VocabularyInfo:
        eos_idx = self._tokenizer.convert_tokens_to_ids(self._eos_token)
        vocab_info = VocabularyInfo(
            size=len(self._tokenizer),
            bos_idx=None,
            eos_idx=eos_idx,
            unk_idx=None,
            pad_idx=None,
        )
        return vocab_info


QWEN_TOKENIZER_FAMILY: Final = "qwen"


def load_qwen_tokenizer(path: Path, card: AssetCard) -> TextTokenizer:
    try:
        return QwenTokenizerHuggingFace(path)
    except ValueError as ex:
        raise TextTokenizerLoadError(
            card.name, f"The '{card.name}' asset card does not contain a valid text tokenizer configuration of the '{QWEN_TOKENIZER_FAMILY}' family. See the nested exception for details."  # fmt: skip
        ) from ex
    except RuntimeError as ex:
        raise TextTokenizerLoadError(
            card.name, f"The '{card.name}' text tokenizer cannot be loaded. See the nested exception for details."  # fmt: skip
        ) from ex
