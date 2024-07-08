# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Sequence, TypeVar, Union, final

from torch import Tensor

from fairseq2.assets import (
    AssetCard,
    AssetCardError,
    AssetDownloadManager,
    AssetError,
    AssetStore,
    default_asset_store,
    default_download_manager,
)
from fairseq2.assets.utils import retrieve_asset_card
from fairseq2.data.vocabulary_info import VocabularyInfo
from fairseq2.typing import Device, override


class TextTokenizer(ABC):
    """Represents a tokenizer to encode and decode text."""

    @abstractmethod
    def create_encoder(
        self,
        *,
        task: Optional[str] = None,
        lang: Optional[str] = None,
        mode: Optional[str] = None,
        device: Optional[Device] = None,
        pin_memory: bool = False,
    ) -> TextTokenEncoder:
        """Create a token encoder.

        The valid arguments for the ``task``, ``lang``, and ``mode`` parameters
        are implementation specific. Refer to concrete ``TextTokenizer``
        subclasses for more information.

        :param task:
            The task for which to generate token indices. Typically, ``task`` is
            used to distinguish between different tasks such as 'translation' or
            'transcription'.
        :param lang:
            The language of generated token indices. Typically, multilingual
            translation tasks use ``lang`` to distinguish between different
            languages such as 'en-US' or 'de-DE'.
        :param mode:
            The mode in which to generate token indices. Typically, translation
            tasks use ``mode`` to distinguish between different modes such as
            'source' or 'target'.
        :param device:
            The device on which to construct tensors.
        :param pin_memory:
            If ``True``, uses pinned memory while constructing tensors.
        """

    @abstractmethod
    def create_raw_encoder(
        self, *, device: Optional[Device] = None, pin_memory: bool = False
    ) -> TextTokenEncoder:
        """Create a raw token encoder with no control symbols.

        :param device:
            The device on which to construct tensors.
        :param pin_memory:
            If ``True``, uses pinned memory while constructing tensors.
        """

    @abstractmethod
    def create_decoder(self) -> TextTokenDecoder:
        """Create a token decoder."""

    @property
    @abstractmethod
    def vocab_info(self) -> VocabularyInfo:
        """The vocabulary information associated with the tokenizer."""


class AbstractTextTokenizer(TextTokenizer):
    """Provides a skeletal implementation of :class:`TextTokenizer`."""

    _vocab_info: VocabularyInfo

    def __init__(self, vocab_info: VocabularyInfo) -> None:
        """
        :param vocab_info:
            The vocabulary information associated with the tokenizer.
        """
        self._vocab_info = vocab_info

    @final
    @property
    @override
    def vocab_info(self) -> VocabularyInfo:
        """The vocabulary information associated with the tokenizer."""
        return self._vocab_info


class TextTokenEncoder(ABC):
    """Encodes text into tokens or token indices."""

    @abstractmethod
    def __call__(self, text: str) -> Tensor:
        """
        :param text:
            The text to encode.
        """

    @abstractmethod
    def encode_as_tokens(self, text: str) -> List[str]:
        """
        :param text:
            The text to encode.
        """

    @property
    @abstractmethod
    def prefix_indices(self) -> Optional[Tensor]:
        """Get the indices of the prefix tokens. *Shape:* :math:`(S)`, where
        :math:`S` is the number of indices."""

    @property
    @abstractmethod
    def suffix_indices(self) -> Optional[Tensor]:
        """Get the indices of the suffix tokens. *Shape:* :math:`(S)`, where
        :math:`S` is the number of indices."""


class TextTokenDecoder(ABC):
    """Decodes text from tokens or token indices."""

    @abstractmethod
    def __call__(self, token_indices: Tensor) -> str:
        """
        :param token_indices:
            The token indices to decode from.
        """

    @abstractmethod
    def decode_from_tokens(self, tokens: Sequence[str]) -> str:
        """
        :param tokens:
            The tokens to decode from.
        """


TextTokenizerT = TypeVar("TextTokenizerT", bound=TextTokenizer)

TextTokenizerT_co = TypeVar("TextTokenizerT_co", bound=TextTokenizer, covariant=True)


class TextTokenizerLoader(Protocol[TextTokenizerT_co]):
    """Loads text tokenizers of type ``TextTokenizerT``."""

    def __call__(
        self,
        tokenizer_name_or_card: Union[str, AssetCard, Path],
        *,
        force: bool = False,
        progress: bool = True,
    ) -> TextTokenizerT_co:
        """
        :param tokenizer_name_or_card:
            The name, asset card, or path to the asset card file of the
            tokenizer to load.
        :param force:
            If ``True``, downloads the tokenizer even if it is already in cache.
        :param progress:
            If ``True``, displays a progress bar to stderr.
        """


class AbstractTextTokenizerLoader(ABC, TextTokenizerLoader[TextTokenizerT]):
    """Provides a skeletal implementation of :class:`TextTokenizerLoader`."""

    _asset_store: AssetStore
    _download_manager: AssetDownloadManager

    def __init__(
        self,
        *,
        asset_store: Optional[AssetStore] = None,
        download_manager: Optional[AssetDownloadManager] = None,
    ) -> None:
        """
        :param asset_store:
            The asset store where to check for available tokenizers. If ``None``,
            the default asset store will be used.
        :param download_manager:
            The download manager. If ``None``, the default download manager will
            be used.
        """
        self._asset_store = asset_store or default_asset_store
        self._download_manager = download_manager or default_download_manager

    @final
    def __call__(
        self,
        tokenizer_name_or_card: Union[str, AssetCard, Path],
        *,
        force: bool = False,
        progress: bool = True,
    ) -> TextTokenizerT:
        card = retrieve_asset_card(tokenizer_name_or_card, self._asset_store)

        tokenizer_ref = card.field("tokenizer_ref").get_as_(str)
        if tokenizer_ref is not None:
            return self(tokenizer_ref, force=force, progress=progress)

        tokenizer_uri = card.field("tokenizer").as_uri()

        try:
            path = self._download_manager.download_tokenizer(
                tokenizer_uri, card.name, force=force, progress=progress
            )
        except ValueError as ex:
            raise AssetCardError(
                f"The value of the field 'tokenizer' of the asset card '{card.name}' must be a URI. See nested exception for details."
            ) from ex

        try:
            return self._load(path, card)
        except ValueError as ex:
            raise AssetError(
                f"The {card.name} tokenizer cannot be loaded. See nested exception for details."
            ) from ex

    @abstractmethod
    def _load(self, path: Path, card: AssetCard) -> TextTokenizerT:
        """
        :param path:
            The path to the tokenizer.
        :param card:
            The asset card of the tokenizer.
        """


@final
class DelegatingTextTokenizerLoader(TextTokenizerLoader[TextTokenizerT]):
    """Loads text tokenizers of type ``TextTokenizerT`` using registered loaders."""

    _asset_store: AssetStore
    _loaders: Dict[str, TextTokenizerLoader[TextTokenizerT]]

    def __init__(self, *, asset_store: Optional[AssetStore] = None) -> None:
        """
        :param asset_store:
            The asset store where to check for available tokenizers. If ``None``,
            the default asset store will be used.
        """
        self._asset_store = asset_store or default_asset_store

        self._loaders = {}

    def __call__(
        self,
        tokenizer_name_or_card: Union[str, AssetCard, Path],
        *,
        force: bool = False,
        progress: bool = True,
    ) -> TextTokenizerT:
        card = retrieve_asset_card(tokenizer_name_or_card, self._asset_store)

        ref = card.field("tokenizer_ref").get_as_(str)
        if ref is not None:
            return self(ref, force=force, progress=progress)

        family = card.field("tokenizer_family").as_(str)

        try:
            loader = self._loaders[family]
        except KeyError:
            raise AssetError(
                f"The value of the field 'tokenizer_family' of the asset card '{card.name}' must be a supported tokenizer family, but '{family}' has no registered loader."
            )

        return loader(card, force=force, progress=progress)

    def register(
        self, family: str, loader: TextTokenizerLoader[TextTokenizerT]
    ) -> None:
        """Register a tokenizer loader to use with this loader.

        :param family:
            The tokenizer family. If the 'tokenizer_family', 'model_family', or
            'dataset_family' field of an asset card matches this value, the
            specified ``loader`` will be used.
        :param loader:
            The tokenizer loader.
        """
        if family in self._loaders:
            raise ValueError(
                f"`family` must be a unique text tokenizer family name, but '{family}' has already a registered loader."
            )

        self._loaders[family] = loader

    def supports(self, tokenizer_name_or_card: Union[str, AssetCard, Path]) -> bool:
        """Return ``True`` if the specified tokenizer has a registered loader."""
        card = retrieve_asset_card(tokenizer_name_or_card, self._asset_store)

        ref = card.field("tokenizer_ref").get_as_(str)
        if ref is not None:
            return self.supports(ref)

        family = card.field("tokenizer_family").as_(str)

        return family in self._loaders


load_text_tokenizer = DelegatingTextTokenizerLoader[TextTokenizer]()
