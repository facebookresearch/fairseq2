# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Sequence,
    TypeVar,
    Union,
    final,
)

from torch import Tensor

from fairseq2.assets import (
    AssetCard,
    AssetCardError,
    AssetCardFieldNotFoundError,
    AssetDownloadManager,
    AssetError,
    AssetStore,
)
from fairseq2.assets import asset_store as default_asset_store
from fairseq2.data.typing import PathLike, StringLike
from fairseq2.data.vocabulary_info import VocabularyInfo
from fairseq2.typing import Device, finaloverride


class TextTokenizer(ABC):
    """Represents a tokenizer to encode and decode text."""

    vocab_info: VocabularyInfo

    def __init__(self, vocab_info: VocabularyInfo) -> None:
        """
        :param vocab_info:
            The vocabulary information associated with the tokenizer.
        """
        self.vocab_info = vocab_info

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
            The task for which to generate token indices. Typically, multi-task
            jobs use ``task`` to distinguish between different tasks such as
            'translation' or 'transcription'.
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


class TextTokenEncoder(ABC):
    """Encodes text into tokens or token indices."""

    @abstractmethod
    def __call__(self, text: StringLike) -> Tensor:
        """
        :param text:
            The text to encode.
        """

    @abstractmethod
    def encode_as_tokens(self, text: StringLike) -> List[StringLike]:
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
    def __call__(self, token_indices: Tensor) -> StringLike:
        """
        :param token_indices:
            The token indices to decode from.
        """

    @abstractmethod
    def decode_from_tokens(self, tokens: Sequence[StringLike]) -> StringLike:
        """
        :param tokens:
            The tokens to decode from.
        """


class TextTokenizerLoader(ABC):
    """Loads text tokenizers."""

    @abstractmethod
    def __call__(
        self,
        tokenizer_name_or_card: Union[str, AssetCard],
        *,
        force: bool = False,
        cache_only: bool = False,
        progress: bool = True,
    ) -> TextTokenizer:
        """
        :param tokenizer_name_or_card:
            The name or asset card of the tokenizer to load.
        :param force:
            If ``True``, downloads the tokenizer even if it is already in cache.
        :param cache_only:
            If ``True``, skips the download and uses the cached tokenizer.
        :param progress:
            If ``True``, displays a progress bar to stderr.
        """


TextTokenizerT = TypeVar("TextTokenizerT", bound=TextTokenizer)
TextTokenizerT_co = TypeVar("TextTokenizerT_co", bound=TextTokenizer, covariant=True)


class GenericTextTokenizerLoader(TextTokenizerLoader, Generic[TextTokenizerT]):
    """Represents an abstract base class for text tokenizer loaders."""

    asset_store: AssetStore
    download_manager: AssetDownloadManager

    def __init__(
        self, asset_store: AssetStore, download_manager: AssetDownloadManager
    ) -> None:
        """
        :param asset_store:
            The asset store where to check for available tokenizers.
        :param download_manager:
            The download manager.
        """
        self.asset_store = asset_store
        self.download_manager = download_manager

    def __call__(
        self,
        tokenizer_name_or_card: Union[str, AssetCard],
        *,
        force: bool = False,
        cache_only: bool = False,
        progress: bool = True,
    ) -> TextTokenizerT:
        if isinstance(tokenizer_name_or_card, AssetCard):
            card = tokenizer_name_or_card
        else:
            card = self.asset_store.retrieve_card(tokenizer_name_or_card)

        uri = card.field("tokenizer").as_uri()

        try:
            path = self.download_manager.download_tokenizer(
                uri, card.name, force=force, cache_only=cache_only, progress=progress
            )
        except ValueError as ex:
            raise AssetCardError(
                f"The value of the field 'tokenizer' of the asset card '{card.name}' is not valid. See nested exception for details."
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
            The asset card of the associated model.
        """


class BasicTextTokenizerFactory(Protocol[TextTokenizerT_co]):
    """Constructs text tokenizers of type ``TextTokenizerT``."""

    def __call__(self, pathname: PathLike) -> TextTokenizerT_co:
        """
        :param pathname:
            The pathname of the tokenizer.
        """


@final
class BasicTextTokenizerLoader(GenericTextTokenizerLoader[TextTokenizerT]):
    """Loads text tokenizers of type ``TokenizerT``."""

    tokenizer_factory: BasicTextTokenizerFactory[TextTokenizerT]

    def __init__(
        self,
        asset_store: AssetStore,
        download_manager: AssetDownloadManager,
        tokenizer_factory: BasicTextTokenizerFactory[TextTokenizerT],
    ) -> None:
        """
        :param asset_store:
            The asset store where to check for available tokenizers.
        :param download_manager:
            The download manager.
        :param tokenizer_factory:
            The factory to construct tokenizers.
        """
        super().__init__(asset_store, download_manager)

        self.tokenizer_factory = tokenizer_factory

    @finaloverride
    def _load(self, path: Path, card: AssetCard) -> TextTokenizerT:
        return self.tokenizer_factory(path)


@final
class CompositeTextTokenizerLoader(TextTokenizerLoader):
    """Loads text tokenizers using registered loaders."""

    asset_store: AssetStore

    _loaders: Dict[str, TextTokenizerLoader]

    def __init__(self, asset_store: AssetStore) -> None:
        """
        :param asset_store:
            The asset store where to check for available tokenizers.
        """
        self.asset_store = asset_store

        self._loaders = {}

    @finaloverride
    def __call__(
        self,
        tokenizer_name_or_card: Union[str, AssetCard],
        *,
        force: bool = False,
        cache_only: bool = False,
        progress: bool = True,
    ) -> TextTokenizer:
        if isinstance(tokenizer_name_or_card, AssetCard):
            card = tokenizer_name_or_card
        else:
            card = self.asset_store.retrieve_card(tokenizer_name_or_card)

        tokenizer_type = None

        for field in ["tokenizer_type", "model_type", "dataset_type"]:
            try:
                tokenizer_type = card.field(field).as_(str)
            except AssetCardFieldNotFoundError:
                continue

        if tokenizer_type is None:
            raise AssetCardFieldNotFoundError(
                f"The asset card '{card.name}' must have a field named 'tokenizer_type', 'model_type', or 'dataset_type'."
            )

        try:
            loader = self._loaders[tokenizer_type]
        except KeyError:
            raise RuntimeError(
                f"The text tokenizer type '{tokenizer_type}' has no registered loader."
            )

        return loader(card, force=force, cache_only=cache_only, progress=progress)

    def register_loader(self, tokenizer_type: str, loader: TextTokenizerLoader) -> None:
        """Register a tokenizer loader to use with this loader.

        :param tokenizer_type:
            The tokenizer type. If the 'tokenizer_type', 'model_type', or
            'dataset_type' field of an asset card matches this value, the
            specified `loader` will be used.
        :param loader:
            The tokenizer loader.
        """
        if tokenizer_type in self._loaders:
            raise ValueError(
                f"`tokenizer_type` must be a unique text tokenizer type, but '{tokenizer_type}' is already registered."
            )

        self._loaders[tokenizer_type] = loader


load_text_tokenizer = CompositeTextTokenizerLoader(default_asset_store)
