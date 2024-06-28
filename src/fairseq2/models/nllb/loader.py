# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import List, final

from fairseq2.assets import AssetCard
from fairseq2.data.text import AbstractTextTokenizerLoader, load_text_tokenizer
from fairseq2.models.nllb.tokenizer import NllbTokenizer
from fairseq2.typing import override


@final
class NllbTokenizerLoader(AbstractTextTokenizerLoader[NllbTokenizer]):
    """Loads NLLB tokenizers."""

    @override
    def _load(self, path: Path, card: AssetCard) -> NllbTokenizer:
        langs = card.field("langs").as_(List[str])

        default_lang = card.field("default_lang").as_(str)

        return NllbTokenizer(path, langs, default_lang)


load_nllb_tokenizer = NllbTokenizerLoader()

load_text_tokenizer.register("nllb", load_nllb_tokenizer)
