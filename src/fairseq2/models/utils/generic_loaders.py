# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from functools import partial
from pathlib import Path
from pickle import PickleError
from typing import Any, Dict, Generic, Optional, Protocol, TypeVar, Union, final

from torch.nn import Module

from fairseq2.assets import (
    AssetCard,
    AssetCardError,
    AssetCardFieldNotFoundError,
    AssetDownloadManager,
    AssetError,
    AssetStore,
)
from fairseq2.data import PathLike
from fairseq2.data.text import TextTokenizer
from fairseq2.models.utils.arch_registry import ArchitectureRegistry
from fairseq2.models.utils.checkpoint import load_checkpoint
from fairseq2.nn.utils.module import (
    infer_device,
    reset_non_persistent_buffers,
    to_empty,
)
from fairseq2.typing import CPU, META, DataType, Device, finaloverride
from fairseq2.utils.dataclass import update_dataclass

logger = logging.getLogger("fairseq2.models")


ConfigT = TypeVar("ConfigT")
ConfigT_contra = TypeVar("ConfigT_contra", contravariant=True)


class ConfigLoader(Generic[ConfigT]):
    """Loads model configurations of type ``ConfigT``."""

    asset_store: AssetStore
    archs: ArchitectureRegistry[ConfigT]

    def __init__(
        self, asset_store: AssetStore, archs: ArchitectureRegistry[ConfigT]
    ) -> None:
        """
        :param asset_store:
            The asset store where to check for available models.
        :param archs:
            The registry containing all supported model architectures.
        """
        self.asset_store = asset_store
        self.archs = archs

    def __call__(self, model_name_or_card: Union[str, AssetCard]) -> ConfigT:
        """
        :param model_name_or_card:
            The name or asset card of the model whose configuration to load.

        :returns:
            The model configuration of ``model_name_or_card``.
        """
        if isinstance(model_name_or_card, AssetCard):
            card = model_name_or_card
        else:
            card = self.asset_store.retrieve_card(model_name_or_card)

        card.field("model_type").check_equals(self.archs.model_type)

        # Ensure that the card has a valid model architecture.
        arch_name = card.field("model_arch").as_one_of(self.archs.names())

        # Load the model configuration.
        config = self.archs.get_config(arch_name)

        # If the card holds a configuration object, it takes precedence.
        try:
            config = card.field("model_config").as_(type(config))

            return deepcopy(config)
        except AssetCardError:
            pass

        # Otherwise, check if we should override anything in the default model
        # configuration.
        try:
            config_overrides = card.field("model_config").as_(dict)
        except AssetCardFieldNotFoundError:
            config_overrides = None

        if config_overrides:
            try:
                update_dataclass(config, deepcopy(config_overrides))
            except (TypeError, ValueError) as ex:
                raise AssetError(
                    f"The model configuration of the asset card '{card.name}' contains one or more invalid keys. See nested exception for details."
                ) from ex

        return config


ModelT = TypeVar("ModelT", bound=Module)
ModelT_co = TypeVar("ModelT_co", bound=Module, covariant=True)


class ModelFactory(Protocol[ConfigT_contra, ModelT_co]):
    """Constructs models of type ``ModelT``."""

    def __call__(
        self,
        config: ConfigT_contra,
        *,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> ModelT_co:
        """
        :param config:
            The model configuration.
        :param device:
            The device on which to initialize the model.
        :param dtype:
            The data type of the model parameters and buffers.
        """


class CheckpointConverter(Protocol[ConfigT_contra]):
    """Converts checkpoints to fairseq2."""

    def __call__(
        self, checkpoint: Dict[str, Any], config: ConfigT_contra
    ) -> Dict[str, Any]:
        """
        :param checkpoint:
            The checkpoint to convert.
        :param config:
            The configuration of the model about to be constructed.

        :returns:
            A converted checkpoint that is compatible with fairseq2.
        """


class ModelLoader(Generic[ModelT, ConfigT]):
    """Loads models of type ``ModelT``."""

    asset_store: AssetStore
    download_manager: AssetDownloadManager
    config_loader: ConfigLoader[ConfigT]
    model_factory: ModelFactory[ConfigT, ModelT]
    checkpoint_converter: Optional[CheckpointConverter[ConfigT]]
    restrict_checkpoints: bool

    def __init__(
        self,
        asset_store: AssetStore,
        download_manager: AssetDownloadManager,
        config_loader: ConfigLoader[ConfigT],
        model_factory: ModelFactory[ConfigT, ModelT],
        checkpoint_converter: Optional[CheckpointConverter[ConfigT]] = None,
        restrict_checkpoints: bool = True,
    ) -> None:
        """
        :param asset_store:
            The asset store where to check for available models.
        :param download_manager:
            The download manager to download model checkpoints.
        :param config_loader:
            The configuration loader.
        :param model_factory:
            The factory to construct models.
        :param checkpoint_converter:
            The converter to which loaded checkpoints will be passed for further
            processing.
        :param restrict_checkpoints:
            If ``True``, restricts the Python unpickler to load only tensors,
            primitive types, and dictionaries.
        """
        self.asset_store = asset_store
        self.download_manager = download_manager
        self.config_loader = config_loader
        self.model_factory = model_factory
        self.checkpoint_converter = checkpoint_converter
        self.restrict_checkpoints = restrict_checkpoints

    def __call__(
        self,
        model_name_or_card: Union[str, AssetCard],
        *,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
        out: Optional[ModelT] = None,
        force: bool = False,
        cache_only: bool = False,
        progress: bool = True,
    ) -> ModelT:
        """
        :param model_name_or_card:
            The name or asset card of the model to load.
        :param device:
            The device on which to load the model.
        :param dtype:
            The data type of the model parameters and buffers.
        :param out:
            The output model to load.
        :param force:
            If ``True``, downloads the model checkpoint even if it is already in
            cache.
        :param cache_only:
            If ``True``, skips the download and uses the cached model checkpoint.
        :param progress:
            If ``True``, displays a progress bar to stderr.

        :returns:
            A model loaded from the checkpoint of ``model_name_or_card``.
        """
        if isinstance(model_name_or_card, AssetCard):
            card = model_name_or_card
        else:
            card = self.asset_store.retrieve_card(model_name_or_card)

        config = self.config_loader(card)

        # Load the checkpoint.
        uri = card.field("checkpoint").as_uri()

        try:
            path = self.download_manager.download_checkpoint(
                uri, card.name, force=force, cache_only=cache_only, progress=progress
            )
        except ValueError as ex:
            raise AssetCardError(
                f"The value of the field 'checkpoint' of the asset card '{card.name}' is not valid. See nested exception for details."
            ) from ex

        if self.checkpoint_converter is None:
            checkpoint_converter = None
        else:
            checkpoint_converter = partial(self.checkpoint_converter, config=config)

        try:
            checkpoint = load_checkpoint(
                path,
                map_location=CPU,
                restrict=self.restrict_checkpoints,
                converter=checkpoint_converter,
            )
        except (RuntimeError, OSError, KeyError, ValueError, PickleError) as ex:
            raise AssetError(
                f"The checkpoint of {card.name} cannot be loaded. See nested exception for details."
            ) from ex

        if out is not None:
            model = out
        else:
            try:
                # Try to construct the model on the meta device.
                model = self.model_factory(config, device=META, dtype=dtype)
            except NotImplementedError:
                logger.warning("One or more operators in %s constructor do not support the meta device. Skipping lazy initialization.", card.name)  # fmt: skip

                # If we are here, it means the model has at least one operator that
                # does not support meta device. Do regular model initialization.
                model = self.model_factory(config, device=device, dtype=dtype)

        model_device = infer_device(model)

        if model_device == META:
            # Move the model to the actual device without initializing. Its
            # state will be overwritten by the checkpoint anyways.
            to_empty(model, device=device or CPU)

        # Load the model.
        try:
            state_dict = checkpoint["model"]
        except KeyError:
            raise AssetError(
                f"The checkpoint of {card.name} does not contain a 'model' entry."
            )

        try:
            model.load_state_dict(state_dict)
        except (KeyError, ValueError) as ex:
            raise AssetError(
                f"{card.name} cannot be loaded. See nested exception for details."
            ) from ex

        if model_device == META:
            # Non-persistent buffers are not included in the checkpoint, so we
            # have to explicitly initialize them.
            reset_non_persistent_buffers(model)

        return model


TokenizerT = TypeVar("TokenizerT", bound=TextTokenizer)
TokenizerT_co = TypeVar("TokenizerT_co", bound=TextTokenizer, covariant=True)


class TokenizerLoaderBase(ABC, Generic[TokenizerT]):
    """Represents an abstract base class for tokenizer loaders."""

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
    ) -> TokenizerT:
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
    def _load(self, path: Path, card: AssetCard) -> TokenizerT:
        """
        :param path:
            The path to the tokenizer.
        :param card:
            The asset card of the associated model.
        """


class TokenizerFactory(Protocol[TokenizerT_co]):
    """Constructs tokenizers of type ``TokenizerT``."""

    def __call__(self, pathname: PathLike) -> TokenizerT_co:
        """
        :param pathname:
            The pathname of the tokenizer.
        """


@final
class TokenizerLoader(TokenizerLoaderBase[TokenizerT]):
    """Loads tokenizers of type ``TokenizerT``."""

    tokenizer_factory: TokenizerFactory[TokenizerT]

    def __init__(
        self,
        asset_store: AssetStore,
        download_manager: AssetDownloadManager,
        tokenizer_factory: TokenizerFactory[TokenizerT],
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
    def _load(self, path: Path, card: AssetCard) -> TokenizerT:
        return self.tokenizer_factory(path)
