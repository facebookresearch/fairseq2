# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Protocol, final

from typing_extensions import override

from fairseq2.assets import AssetCard, AssetDownloadManager
from fairseq2.data.text.tokenizers.tokenizer import TextTokenizer


class TextTokenizerHandler(ABC):
    @abstractmethod
    def load(self, card: AssetCard, *, force: bool = False) -> TextTokenizer:
        ...


class TextTokenizerNotFoundError(LookupError):
    name: str

    def __init__(self, name: str) -> None:
        super().__init__(f"'{name}' is not a known text tokenizer.")

        self.name = name


class TextTokenizerLoader(Protocol):
    def __call__(self, path: Path, card: AssetCard) -> TextTokenizer:
        ...


@final
class StandardTextTokenizerHandler(TextTokenizerHandler):
    _loader: TextTokenizerLoader
    _asset_download_manager: AssetDownloadManager

    def __init__(
        self,
        *,
        loader: TextTokenizerLoader,
        asset_download_manager: AssetDownloadManager,
    ) -> None:
        self._loader = loader
        self._asset_download_manager = asset_download_manager

    @override
    def load(self, card: AssetCard, *, force: bool = False) -> TextTokenizer:
        tokenizer_uri = card.field("tokenizer").as_uri()

        path = self._asset_download_manager.download_tokenizer(
            tokenizer_uri, card.name, force=force
        )

        return self._loader(path, card)


def get_text_tokenizer_family(card: AssetCard) -> str:
    return card.field("tokenizer_family").as_(str)  # type: ignore[no-any-return]
