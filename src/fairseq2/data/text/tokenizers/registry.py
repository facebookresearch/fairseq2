# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Protocol, TypeAlias, final

from typing_extensions import override

from fairseq2.assets import AssetCard, AssetDownloadManager
from fairseq2.data.text.tokenizers.tokenizer import TextTokenizer
from fairseq2.utils.registry import Registry


class TextTokenizerHandler(ABC):
    @abstractmethod
    def load(
        self,
        card: AssetCard,
        asset_download_manager: AssetDownloadManager,
        *,
        force: bool = False,
    ) -> TextTokenizer:
        ...


TextTokenizerRegistry: TypeAlias = Registry[TextTokenizerHandler]


class TextTokenizerLoader(Protocol):
    def __call__(self, path: Path, card: AssetCard) -> TextTokenizer:
        ...


@final
class StandardTextTokenizerHandler(TextTokenizerHandler):
    _loader: TextTokenizerLoader

    def __init__(self, *, loader: TextTokenizerLoader) -> None:
        self._loader = loader

    @override
    def load(
        self,
        card: AssetCard,
        asset_download_manager: AssetDownloadManager,
        *,
        force: bool = False,
    ) -> TextTokenizer:
        tokenizer_uri = card.field("tokenizer").as_uri()

        path = asset_download_manager.download_tokenizer(
            tokenizer_uri, card.name, force=force
        )

        return self._loader(path, card)


def get_text_tokenizer_family(card: AssetCard) -> str:
    return card.field("tokenizer_family").as_(str)  # type: ignore[no-any-return]
