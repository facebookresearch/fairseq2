# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import TypeVar, final

from typing_extensions import override

from fairseq2.data.tokenizers import (
    TokenDecoder,
    TokenEncoder,
    Tokenizer,
    TokenizerFamily,
    VocabularyInfo,
)
from fairseq2.device import Device

TokenizerT = TypeVar("TokenizerT")


@final
class RecipeTokenizer(Tokenizer):
    def __init__(
        self, inner_tokenizer: Tokenizer, config: object, family: TokenizerFamily
    ) -> None:
        self._inner_tokenizer = inner_tokenizer
        self._config = config
        self._family = family

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
        return self._inner_tokenizer.create_encoder(
            task=task, lang=lang, mode=mode, device=device, pin_memory=pin_memory
        )

    @override
    def create_raw_encoder(
        self, *, device: Device | None = None, pin_memory: bool = False
    ) -> TokenEncoder:
        return self._inner_tokenizer.create_raw_encoder(
            device=device, pin_memory=pin_memory
        )

    @override
    def create_decoder(self, *, skip_special_tokens: bool = False) -> TokenDecoder:
        return self._inner_tokenizer.create_decoder(
            skip_special_tokens=skip_special_tokens
        )

    def as_(self, kls: type[TokenizerT]) -> TokenizerT:
        if not isinstance(self._inner_tokenizer, kls):
            raise TypeError(
                f"Tokenizer is expected to be of type `{kls}`, but is of type `{type(self._inner_tokenizer)}` instead."
            )

        return self._inner_tokenizer

    @property
    def config(self) -> object:
        return self._config

    @property
    def family(self) -> TokenizerFamily:
        return self._family

    @property
    @override
    def vocab_info(self) -> VocabularyInfo:
        return self._inner_tokenizer.vocab_info
