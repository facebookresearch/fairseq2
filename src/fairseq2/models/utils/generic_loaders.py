# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
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
from fairseq2.data.text import AbstractTextTokenizerLoader
from fairseq2.data.text.text_tokenizer import TextTokenizer
from fairseq2.models.utils.arch_registry import ArchitectureRegistry
from fairseq2.models.utils.checkpoint import load_checkpoint
from fairseq2.nn.utils.module import (
    infer_device,
    load_state_dict,
    reset_non_persistent_buffers,
    to_empty,
)
from fairseq2.typing import CPU, META, DataType, Device, override
from fairseq2.utils.dataclass import update_dataclass

logger = logging.getLogger("fairseq2.models")


ConfigT = TypeVar("ConfigT")

ConfigT_contra = TypeVar("ConfigT_contra", contravariant=True)

TextTokenizerT = TypeVar("TextTokenizerT", bound=TextTokenizer)

TextTokenizerT_co = TypeVar("TextTokenizerT_co", bound=TextTokenizer, covariant=True)


class TextTokenizerFactory(Protocol[TextTokenizerT_co]):
    """Constructs text tokenizers of type ``TextTokenizerT``."""

    def __call__(self, path: Path, card: AssetCard) -> TextTokenizerT_co:
        """
        :param path:
            The path to the tokenizer.
        :param card:
            The asset card of the tokenizer.
        """


@final
class StandardTextTokenizerLoader(AbstractTextTokenizerLoader[TextTokenizerT]):
    _factory: TextTokenizerFactory[TextTokenizerT]

    def __init__(
        self,
        asset_store: AssetStore,
        download_manager: AssetDownloadManager,
        factory: TextTokenizerFactory[TextTokenizerT],
    ) -> None:
        """
        :param asset_store:
            The asset store where to check for available tokenizers.
        :param download_manager:
            The download manager.
        :param factory:
            The factory to construct tokenizers.
        """
        super().__init__(asset_store, download_manager)

        self._factory = factory

    @override
    def _load(self, path: Path, card: AssetCard) -> TextTokenizerT:
        return self._factory(path, card)


@final
class ConfigLoader(Generic[ConfigT]):
    """Loads model configurations of type ``ConfigT``."""

    _asset_store: AssetStore
    _archs: ArchitectureRegistry[ConfigT]

    def __init__(
        self, asset_store: AssetStore, archs: ArchitectureRegistry[ConfigT]
    ) -> None:
        """
        :param asset_store:
            The asset store where to check for available models.
        :param archs:
            The registry containing all supported model architectures.
        """
        self._asset_store = asset_store
        self._archs = archs

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
            card = self._asset_store.retrieve_card(model_name_or_card)

        card.field("model_type").check_equals(self._archs.model_type)

        # Ensure that the card has a valid model architecture.
        arch_name = card.field("model_arch").as_one_of(self._archs.names())

        # Load the model configuration.
        config = self._archs.get_config(arch_name)

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


@final
class ModelLoader(Generic[ModelT, ConfigT]):
    """Loads models of type ``ModelT``."""

    asset_store: AssetStore
    download_manager: AssetDownloadManager
    config_loader: ConfigLoader[ConfigT]
    model_factory: ModelFactory[ConfigT, ModelT]
    checkpoint_converter: Optional[CheckpointConverter[ConfigT]]
    mmap: bool
    restrict_checkpoints: bool
    skip_meta_init: bool

    def __init__(
        self,
        asset_store: AssetStore,
        download_manager: AssetDownloadManager,
        config_loader: ConfigLoader[ConfigT],
        model_factory: ModelFactory[ConfigT, ModelT],
        checkpoint_converter: Optional[CheckpointConverter[ConfigT]] = None,
        *,
        mmap: bool = False,
        restrict_checkpoints: bool = True,
        skip_meta_init: bool = False,
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
        :param mmap:
            If ``True``, indicates whether the checkpoint should be memory
            mapped.
        :param restrict_checkpoints:
            If ``True``, restricts the Python unpickler to load only tensors,
            primitive types, and dictionaries.
        :param skip_meta_init:
            If ``True``, skips meta device initialization and constructs the
            model directly on the requested device. Meant to be used with models
            that do not support PyTorch's ``reset_parameters()`` convention.
        """
        self.asset_store = asset_store
        self.download_manager = download_manager
        self.config_loader = config_loader
        self.model_factory = model_factory
        self.checkpoint_converter = checkpoint_converter
        self.mmap = mmap
        self.restrict_checkpoints = restrict_checkpoints
        self.skip_meta_init = skip_meta_init

    def __call__(
        self,
        model_name_or_card: Union[str, AssetCard],
        *,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
        out: Optional[ModelT] = None,
        force: bool = False,
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
                uri, card.name, force=force, progress=progress
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
                mmap=self.mmap,
                restrict=self.restrict_checkpoints,
                converter=checkpoint_converter,
            )
        except (RuntimeError, OSError, KeyError, ValueError, PickleError) as ex:
            raise AssetError(
                f"The checkpoint of {card.name} cannot be loaded. See nested exception for details."
            ) from ex

        if out is not None:
            model = out

            model_device = infer_device(model, name="out")
        else:
            if self.skip_meta_init:
                model = self.model_factory(config, device=device, dtype=dtype)
            else:
                try:
                    # Try to construct the model on the meta device.
                    model = self.model_factory(config, device=META, dtype=dtype)
                except NotImplementedError:
                    logger.warning("One or more operators in %s constructor do not support the meta device. Skipping meta device initialization.", card.name)  # fmt: skip

                    # If we are here, it means the model constructor has at
                    # least one operation that does not support the meta device.
                    # Fall back to regular model initialization.
                    model = self.model_factory(config, device=device, dtype=dtype)

                try:
                    model_device = infer_device(model, name="model")
                except ValueError as ex:
                    raise RuntimeError(
                        "`model_factory` returned a model that is not constructed correctly. See nested exception for details."
                    ) from ex

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
            load_state_dict(model, state_dict)
        except (KeyError, ValueError) as ex:
            raise AssetError(
                f"{card.name} cannot be loaded. See nested exception for details."
            ) from ex

        if model_device == META:
            # Non-persistent buffers are not included in the checkpoint, so we
            # have to explicitly initialize them.
            reset_non_persistent_buffers(model)

        return model
