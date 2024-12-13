# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, TypeVar, cast, final

from torch import Tensor
from typing_extensions import override

from fairseq2.assets import (
    AssetCard,
    AssetCardError,
    AssetDownloadManager,
    AssetError,
    AssetStore,
    default_asset_download_manager,
    default_asset_store,
)
from fairseq2.data.vocabulary_info import VocabularyInfo
from fairseq2.typing import Device


class TextTokenizer(ABC):
    """Represents a tokenizer to encode and decode text."""

    @abstractmethod
    def create_encoder(
        self,
        *,
        task: str | None = None,
        lang: str | None = None,
        mode: str | None = None,
        device: Device | None = None,
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
        self, *, device: Device | None = None, pin_memory: bool = False
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
    def encode_as_tokens(self, text: str) -> list[str]:
        """
        :param text:
            The text to encode.
        """

    @property
    @abstractmethod
    def prefix_indices(self) -> Tensor | None:
        """Get the indices of the prefix tokens. *Shape:* :math:`(S)`, where
        :math:`S` is the number of indices."""

    @property
    @abstractmethod
    def suffix_indices(self) -> Tensor | None:
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
        tokenizer_name_or_card: str | AssetCard,
        *,
        force: bool = False,
        progress: bool = True,
    ) -> TextTokenizerT_co:
        """
        :param tokenizer_name_or_card:
            The name or the asset card of the tokenizer to load.
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
        asset_store: AssetStore | None = None,
        download_manager: AssetDownloadManager | None = None,
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
        self._download_manager = download_manager or default_asset_download_manager

    @final
    def __call__(
        self,
        tokenizer_name_or_card: str | AssetCard,
        *,
        force: bool = False,
        progress: bool = True,
    ) -> TextTokenizerT:
        if isinstance(tokenizer_name_or_card, AssetCard):
            card = tokenizer_name_or_card
        else:
            card = self._asset_store.retrieve_card(tokenizer_name_or_card)

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
    _loaders: dict[str, TextTokenizerLoader[TextTokenizerT]]

    def __init__(self, *, asset_store: AssetStore | None = None) -> None:
        """
        :param asset_store:
            The asset store where to check for available tokenizers. If ``None``,
            the default asset store will be used.
        """
        self._asset_store = asset_store or default_asset_store

        self._loaders = {}

    def __call__(
        self,
        tokenizer_name_or_card: str | AssetCard,
        *,
        force: bool = False,
        progress: bool = True,
    ) -> TextTokenizerT:
        if isinstance(tokenizer_name_or_card, AssetCard):
            card = tokenizer_name_or_card
        else:
            card = self._asset_store.retrieve_card(tokenizer_name_or_card)

        ref = card.field("tokenizer_ref").get_as_(str)
        if ref is not None:
            return self(ref, force=force, progress=progress)

        family = card.field("tokenizer_family").as_(str)

        try:
            loader = self._loaders[family]
        except KeyError:
            raise AssetError(
                f"The value of the field 'tokenizer_family' of the asset card '{card.name}' must be a supported tokenizer family, but is '{family}' instead."
            ) from None

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
                f"`family` must be a unique text tokenizer family name, but '{family}' is already registered."
            )

        self._loaders[family] = loader

    def supports(self, tokenizer_name_or_card: str | AssetCard) -> bool:
        """Return ``True`` if the specified tokenizer has a registered loader."""
        if isinstance(tokenizer_name_or_card, AssetCard):
            card = tokenizer_name_or_card
        else:
            card = self._asset_store.retrieve_card(tokenizer_name_or_card)

        ref = card.field("tokenizer_ref").get_as_(str)
        if ref is not None:
            return self.supports(ref)

        family = card.field("tokenizer_family").as_(str)

        return family in self._loaders


load_text_tokenizer = DelegatingTextTokenizerLoader[TextTokenizer]()


def is_tokenizer_card(card: AssetCard) -> bool:
    """Return ``True`` if ``card`` specifies a tokenizer."""
    return card.field("tokenizer_family").exists()


def get_tokenizer_family(card: AssetCard) -> str:
    """Return the tokenizer family name contained in ``card``."""
    return cast(str, card.field("tokenizer_family").as_(str))
