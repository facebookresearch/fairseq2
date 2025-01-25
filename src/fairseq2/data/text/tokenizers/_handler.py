# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import final

from typing_extensions import override

from fairseq2.assets import AssetCard, AssetCardError, AssetDownloadManager
from fairseq2.data.text.tokenizers._error import text_tokenizer_asset_card_error
from fairseq2.data.text.tokenizers._tokenizer import TextTokenizer


class TextTokenizerHandler(ABC):
    @abstractmethod
    def load(self, card: AssetCard) -> TextTokenizer: ...

    @property
    @abstractmethod
    def family(self) -> str: ...


class AbstractTextTokenizerHandler(TextTokenizerHandler):
    _asset_download_manager: AssetDownloadManager

    def __init__(self, asset_download_manager: AssetDownloadManager) -> None:
        self._asset_download_manager = asset_download_manager

    @final
    @override
    def load(self, card: AssetCard) -> TextTokenizer:
        tokenizer_name = card.name

        try:
            tokenizer_uri = card.field("tokenizer").as_uri()
        except AssetCardError as ex:
            raise text_tokenizer_asset_card_error(tokenizer_name) from ex

        path = self._asset_download_manager.download_tokenizer(
            tokenizer_uri, tokenizer_name
        )

        return self._load_tokenizer(path, card)

    @abstractmethod
    def _load_tokenizer(self, path: Path, card: AssetCard) -> TextTokenizer: ...
