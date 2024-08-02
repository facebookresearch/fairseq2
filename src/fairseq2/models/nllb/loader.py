# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from pathlib import Path
from typing import final

from typing_extensions import override

from fairseq2.assets import AssetCard
from fairseq2.data.text import AbstractTextTokenizerLoader, load_text_tokenizer
from fairseq2.models.nllb.tokenizer import NllbTokenizer


@final
class NllbTokenizerLoader(AbstractTextTokenizerLoader[NllbTokenizer]):
    """Loads NLLB tokenizers."""

    @override
    def _load(self, path: Path, card: AssetCard) -> NllbTokenizer:
        langs = card.field("langs").as_(list[str])

        default_lang = card.field("default_lang").as_(str)

        return NllbTokenizer(path, langs, default_lang)


load_nllb_tokenizer = NllbTokenizerLoader()

load_text_tokenizer.register("nllb", load_nllb_tokenizer)
