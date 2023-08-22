# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
from functools import partial
from typing import (
    Any,
    Generic,
    Mapping,
    MutableMapping,
    Optional,
    Protocol,
    TypeVar,
    Union,
)

from torch.nn import Module

from fairseq2.assets import (
    AssetCard,
    AssetCardFieldNotFoundError,
    AssetDownloadManager,
    AssetError,
    AssetStore,
)
from fairseq2.models.utils.arch_registry import ArchitectureRegistry
from fairseq2.models.utils.checkpoint_loader import load_checkpoint
from fairseq2.nn.utils.module import reset_non_persistent_buffers
from fairseq2.typing import DataType, Device
from fairseq2.utils.dataclass import update_dataclass

ModelT = TypeVar("ModelT", bound=Module)

ModelT_co = TypeVar("ModelT_co", bound=Module, covariant=True)

ModelConfigT = TypeVar("ModelConfigT")

ModelConfigT_contra = TypeVar("ModelConfigT_contra", contravariant=True)


class ModelConfigLoader(Generic[ModelConfigT]):
    """Loads model configurations of type ``ModelConfigT``."""

    asset_store: AssetStore
    archs: ArchitectureRegistry[ModelConfigT]

    def __init__(
        self, asset_store: AssetStore, archs: ArchitectureRegistry[ModelConfigT]
    ) -> None:
        """
        :param asset_store:
            The asset store where to check for available models.
        :param archs:
            The registry containing all supported model architectures.
        """
        self.asset_store = asset_store
        self.archs = archs

    def __call__(self, model_name_or_card: Union[str, AssetCard]) -> ModelConfigT:
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

        try:
            config_overrides = card.field("model_config").as_(MutableMapping[str, Any])
        except AssetCardFieldNotFoundError:
            config_overrides = None

        if config_overrides:
            try:
                update_dataclass(config, deepcopy(config_overrides))
            except ValueError as ex:
                raise AssetError(
                    f"The model configuration of the asset card '{card.name}' contains one or more invalid keys. See nested exception for details."
                ) from ex

        return config


class ModelFactory(Protocol[ModelConfigT_contra, ModelT_co]):
    """Constructs models of type ``ModelT``."""

    def __call__(
        self,
        config: ModelConfigT_contra,
        device: Optional[Device],
        dtype: Optional[DataType],
    ) -> ModelT_co:
        """
        :param config:
            The model configuration to use.
        :param device:
            The device on which to initialize the model.
        :param dtype:
            The data type of the model parameters and buffers.
        """


class ModelLoader(Generic[ModelT, ModelConfigT]):
    """Loads models of type ``ModelT``."""

    asset_store: AssetStore
    download_manager: AssetDownloadManager
    model_factory: ModelFactory[ModelConfigT, ModelT]
    config_loader: ModelConfigLoader[ModelConfigT]

    def __init__(
        self,
        asset_store: AssetStore,
        download_manager: AssetDownloadManager,
        model_factory: ModelFactory[ModelConfigT, ModelT],
        archs: ArchitectureRegistry[ModelConfigT],
    ) -> None:
        """
        :param asset_store:
            The asset store where to check for available models.
        :param download_manager:
            The download manager to use to download model checkpoints.
        :param model_factory:
            The callable responsible for constructing models.
        :param archs:
            The registry containing all supported model architectures.
        """
        self.asset_store = asset_store
        self.download_manager = download_manager
        self.model_factory = model_factory

        self.config_loader = ModelConfigLoader(asset_store, archs)

    def __call__(
        self,
        model_name_or_card: Union[str, AssetCard],
        force: bool = False,
        progress: bool = True,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> ModelT:
        """
        :param model_name_or_card:
            The name or asset card of the model to load.
        :param force:
            If ``True``, downloads the checkpoint even if it is already in cache.
        :param progress:
            If ``True``, displays a progress bar to stderr.
        :param device:
            The device on which to load the model.
        :param dtype:
            The data type of the model parameters and buffers.

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

        pathname = self.download_manager.download_checkpoint(
            uri, card.name, force=force, progress=progress
        )

        checkpoint = load_checkpoint(
            pathname,
            card.name,
            map_location="cpu",
            converter=partial(self._upgrade_checkpoint, config=config),
        )

        try:
            # Try to construct the model on the meta device.
            model = self.model_factory(config, Device("meta"), dtype)
        except NotImplementedError:
            # If we are here, it means the model has at least one operator that
            # does not support meta device. Do regular model initialization.
            model = self.model_factory(config, device, dtype)
        else:
            # Move the model to the actual device without initializing. Its
            # state will be overwritten by the checkpoint anyways.
            model = model.to_empty(device=device or "cpu")

        # Load the model.
        try:
            state_dict = checkpoint["model"]
        except KeyError:
            raise AssetError(
                f"The checkpoint of the model '{card.name}' does not contain a 'model' entry."
            )

        try:
            model.load_state_dict(state_dict)
        except (KeyError, ValueError) as ex:
            raise AssetError(
                f"The checkpoint of the model '{card.name}' cannot be loaded. See nested exception for details."
            ) from ex

        # Non-persistent buffers are not included in the checkpoint, so we have
        # to explicitly initialize them.
        reset_non_persistent_buffers(model)

        return model

    def _upgrade_checkpoint(
        self, checkpoint: Mapping[str, Any], config: ModelConfigT
    ) -> Mapping[str, Any]:
        """Upgrade ``checkpoint`` to be compatible with fairseq2.

        :param checkpoint:
            The legacy checkpoint.
        :param config:
            The configuration of the model about to be constructed.

        :returns:
            A checkpoint that is compatible with fairseq2.
        """
        return checkpoint
