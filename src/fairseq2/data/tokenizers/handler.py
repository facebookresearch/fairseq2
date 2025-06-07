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

from fairseq2.assets import AssetCard, AssetCardError, AssetDownloadManager
from fairseq2.data.tokenizers.error import tokenizer_asset_card_error
from fairseq2.data.tokenizers.tokenizer import Tokenizer
from fairseq2.dependency import DependencyContainer, DependencyResolver


class TokenizerHandler(ABC):
    @abstractmethod
    def load(self, resolver: DependencyResolver, card: AssetCard) -> Tokenizer: ...

    @property
    @abstractmethod
    def family(self) -> str: ...


class TokenizerLoader(Protocol):
    def __call__(
        self, resolver: DependencyResolver, path: Path, card: AssetCard
    ) -> Tokenizer: ...


@final
class DelegatingTokenizerHandler(TokenizerHandler):
    _family: str
    _loader: TokenizerLoader
    _asset_download_manager: AssetDownloadManager

    def __init__(
        self,
        family: str,
        loader: TokenizerLoader,
        asset_download_manager: AssetDownloadManager,
    ) -> None:
        self._family = family
        self._loader = loader
        self._asset_download_manager = asset_download_manager

    @override
    def load(self, resolver: DependencyResolver, card: AssetCard) -> Tokenizer:
        name = card.name

        try:
            uri = card.field("tokenizer").as_uri()
        except AssetCardError as ex:
            raise tokenizer_asset_card_error(name) from ex

        path = self._asset_download_manager.download_tokenizer(uri, name)

        return self._loader(resolver, path, card)

    @property
    @override
    def family(self) -> str:
        return self._family


def register_tokenizer_family(
    container: DependencyContainer, family: str, loader: TokenizerLoader
) -> None:
    def create_handler(resolver: DependencyResolver) -> TokenizerHandler:
        asset_download_manager = resolver.resolve(AssetDownloadManager)

        return DelegatingTokenizerHandler(family, loader, asset_download_manager)

    container.register(TokenizerHandler, create_handler, key=family)
