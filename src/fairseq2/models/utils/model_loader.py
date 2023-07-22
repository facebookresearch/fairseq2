# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
from functools import partial
from typing import Any, Generic, Mapping, MutableMapping, Optional, Protocol, TypeVar

from torch.nn import Module

from fairseq2.assets import (
    AssetCardFieldNotFoundError,
    AssetDownloadManager,
    AssetError,
    AssetStore,
)
from fairseq2.models.utils.arch import ArchitectureRegistry
from fairseq2.models.utils.checkpoint import MapLocation, load_checkpoint
from fairseq2.typing import DataType, Device
from fairseq2.utils.dataclass import update_dataclass

ModelT = TypeVar("ModelT", bound=Module, covariant=True)

ModelConfigT = TypeVar("ModelConfigT", contravariant=True)


class ModelFactory(Protocol[ModelConfigT, ModelT]):
    """Constructs models of type ``ModelT``."""

    def __call__(
        self, config: ModelConfigT, device: Optional[Device], dtype: Optional[DataType]
    ) -> ModelT:
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
        self.archs = archs

    def __call__(
        self,
        model_name: str,
        force: bool = False,
        progress: bool = True,
        map_location: Optional[MapLocation] = None,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> ModelT:
        """
        :param model_name:
            The name of the model to load.
        :param force:
            If ``True``, downloads the checkpoint even if it is already in cache.
        :param progress:
            If ``True``, displays a progress bar to stderr.
        :param map_location:
            See `map_location` in :func:`torch.load`.
        :param device:
            The device on which to load the model.
        :param dtype:
            The data type of the model parameters and buffers.

        :returns:
            A model loaded from the checkpoint of ``model_name``.
        """
        card = self.asset_store.retrieve_card(model_name)

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

        # Load the checkpoint.
        uri = card.field("checkpoint").as_uri()

        pathname = self.download_manager.download_checkpoint(
            uri, card.name, force=force, progress=progress
        )

        checkpoint = load_checkpoint(
            pathname,
            card.name,
            map_location=map_location,
            converter=partial(self._upgrade_checkpoint, config=config),
        )

        # Construct and load the model.
        model = self.model_factory(config, device, dtype)

        try:
            state_dict = checkpoint["model"]
        except KeyError:
            raise AssetError(
                f"The checkpoint of the model '{model_name}' does not contain a 'model' entry."
            )

        try:
            model.load_state_dict(state_dict)
        except (KeyError, ValueError) as ex:
            raise AssetError(
                f"The checkpoint of the model '{model_name}' cannot be loaded. See nested exception for details."
            ) from ex

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
