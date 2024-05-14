# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from pickle import PickleError
from typing import Any, Dict, Generic, Optional, Protocol, TypeVar, Union, final

from torch.nn import Module
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

from fairseq2.assets import (
    AssetCard,
    AssetCardError,
    AssetDownloadManager,
    AssetError,
    AssetStore,
    default_asset_store,
    default_download_manager,
)
from fairseq2.gang import Gang
from fairseq2.logging import get_log_writer
from fairseq2.models.config_loader import ModelConfigLoader
from fairseq2.models.utils.checkpoint import load_checkpoint
from fairseq2.nn.utils.module import (
    infer_device,
    load_state_dict,
    reset_non_persistent_buffers,
    to_empty,
)
from fairseq2.typing import CPU, META, DataClass, DataType, Device

log = get_log_writer(__name__)


ModelT = TypeVar("ModelT", bound=Module)

ModelT_co = TypeVar("ModelT_co", bound=Module, covariant=True)

ModelConfigT = TypeVar("ModelConfigT", bound=DataClass)

ModelConfigT_contra = TypeVar(
    "ModelConfigT_contra", bound=DataClass, contravariant=True
)


class ModelLoader(Protocol[ModelT_co]):
    """Loads models of type ``ModelT``."""

    def __call__(
        self,
        model_name_or_card: Union[str, AssetCard],
        *,
        gangs: Optional[Dict[str, Gang]] = None,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
        force: bool = False,
        progress: bool = True,
    ) -> ModelT_co:
        """
        :param model_name_or_card:
            The name or asset card of the model to load.
        :param gangs:
            The gangs over which to shard the model (e.g. for tensor or pipeline
            parallelism).
        :param device:
            The device on which to load the model.
        :param dtype:
            The data type of the model parameters and buffers.
        :param force:
            If ``True``, downloads the model checkpoint even if it is already in
            cache.
        :param progress:
            If ``True``, displays a progress bar to stderr.

        :returns:
            A model loaded from the checkpoint of ``model_name_or_card``.
        """


class CheckpointConverter(Protocol[ModelConfigT_contra]):
    """Converts checkpoints to fairseq2 format."""

    def __call__(
        self, checkpoint: Dict[str, Any], config: ModelConfigT_contra
    ) -> Dict[str, Any]:
        """
        :param checkpoint:
            The checkpoint to convert.
        :param config:
            The configuration of the model about to be constructed.
        """


class DenseModelFactory(Protocol[ModelConfigT_contra, ModelT_co]):
    """Constructs dense models of type ``ModelT``."""

    def __call__(
        self,
        config: ModelConfigT_contra,
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


class DenseModelLoader(ModelLoader[ModelT], Generic[ModelT, ModelConfigT]):
    """Loads dense models of type ``ModelT``."""

    _asset_store: AssetStore
    _download_manager: AssetDownloadManager
    _config_loader: ModelConfigLoader[ModelConfigT]
    _factory: DenseModelFactory[ModelConfigT, ModelT]
    _checkpoint_converter: Optional[CheckpointConverter[ModelConfigT]]
    _mmap: bool
    _restrict_checkpoints: bool
    _skip_meta_init: bool

    def __init__(
        self,
        *,
        config_loader: ModelConfigLoader[ModelConfigT],
        factory: DenseModelFactory[ModelConfigT, ModelT],
        checkpoint_converter: Optional[CheckpointConverter[ModelConfigT]] = None,
        mmap: bool = False,
        restrict_checkpoints: bool = True,
        skip_meta_init: bool = False,
        asset_store: Optional[AssetStore] = None,
        download_manager: Optional[AssetDownloadManager] = None,
    ) -> None:
        """
        :param config_loader:
            The configuration loader.
        :param factory:
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
            model directly on the requested device. Should be used with models
            that do not support PyTorch's ``reset_parameters()`` convention.
        :param asset_store:
            The asset store where to check for available models.
        :param download_manager:
            The download manager to download model checkpoints.
        """
        self._asset_store = asset_store or default_asset_store
        self._download_manager = download_manager or default_download_manager
        self._config_loader = config_loader
        self._factory = factory
        self._checkpoint_converter = checkpoint_converter
        self._mmap = mmap
        self._restrict_checkpoints = restrict_checkpoints
        self._skip_meta_init = skip_meta_init

    @final
    def __call__(
        self,
        model_name_or_card: Union[str, AssetCard],
        *,
        gangs: Optional[Dict[str, Gang]] = None,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
        force: bool = False,
        progress: bool = True,
    ) -> ModelT:
        # Retrieve the gang for tensor parallelism.
        gang = gangs.get("tp") if gangs is not None else None

        if gang is not None:
            if device is None:
                device = gang.device
            elif device != gang.device and device != META:
                raise ValueError(
                    "`device` must either match `gang['tp'].size` or must be of type `meta`."
                )

        if device is None:
            device = CPU

        if isinstance(model_name_or_card, AssetCard):
            card = model_name_or_card
        else:
            card = self._asset_store.retrieve_card(model_name_or_card)

        num_shards = card.field("num_shards").get_as_(int, default=1)
        if num_shards < 1:
            raise AssetCardError(
                f"The value of the field 'num_shards' of the asset card '{card.name}' must be greater than or equal to 1, but is {num_shards} instead."
            )

        if num_shards > 1:
            if gang is None:
                raise ValueError(
                    f"`gangs['tp']` must be specified since {card.name} has {num_shards} checkpoint shards."
                )

            if gang.size != num_shards:
                raise ValueError(
                    f"`gangs['tp'].size` must match the number of checkpoint shards of {card.name} ({num_shards}), but is {gang.size} instead."
                )
        else:
            if gang is not None and gang.size > 1:
                raise ValueError(
                    f"`gangs['tp'].size` must be 1 since the checkpoint of {card.name} is not sharded, but is {gang.size} instead."
                )

        model = None

        config = self._config_loader(card)

        if device == META:
            model = self._factory(config, device=META, dtype=dtype)

            if gang is not None and gang.size > 1:
                self._shard(model, gangs, card)  # type: ignore[arg-type]

            return model

        # Load the checkpoint.
        uri = card.field("checkpoint").as_uri()

        shard_idx = gang.rank if gang is not None and gang.size != 1 else None

        try:
            path = self._download_manager.download_checkpoint(
                uri, card.name, shard_idx=shard_idx, force=force, progress=progress
            )
        except ValueError as ex:
            raise AssetCardError(
                f"The value of the field 'checkpoint' of the asset card '{card.name}' must be URI. See nested exception for details."
            ) from ex

        if self._checkpoint_converter is None:
            checkpoint_converter = None
        else:
            checkpoint_converter = partial(self._checkpoint_converter, config=config)

        try:
            checkpoint = load_checkpoint(
                path,
                map_location=CPU,
                mmap=self._mmap,
                restrict=self._restrict_checkpoints,
                converter=checkpoint_converter,
            )
        except (RuntimeError, OSError, KeyError, ValueError, PickleError) as ex:
            raise AssetError(
                f"The checkpoint of {card.name} cannot be loaded. See nested exception for details."
            ) from ex

        if not self._skip_meta_init:
            try:
                # Try to construct the model on the meta device.
                model = self._factory(config, device=META, dtype=dtype)
            except NotImplementedError:
                log.warning("One or more operators in {} constructor do not support the meta device. Skipping meta device initialization.", card.name)  # fmt: skip

        if model is None:
            # Load on CPU to avoid OOM errors. This is particularly relevant for
            # models that require sharding after initialization.
            model = self._factory(config, device=CPU, dtype=dtype)

        if gang is not None and gang.size > 1:
            self._shard(model, gangs, card)  # type: ignore[arg-type]

        try:
            model_device = infer_device(model, name="model")
        except ValueError as ex:
            raise RuntimeError(
                "`factory` returned a model that is not constructed correctly. See nested exception for details."
            ) from ex

        if model_device != device:
            # Move the model to the actual device without initializing. Its
            # state will be overwritten by the checkpoint anyways.
            to_empty(model, device=device)

        # Load the model.
        try:
            state_dict = checkpoint["model"]
        except KeyError:
            raise AssetError(
                f"The checkpoint of {card.name} does not contain a 'model' entry."
            )

        # Remove DDP 'module' prefix.
        consume_prefix_in_state_dict_if_present(state_dict, prefix="module.")

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

    def _shard(self, model: ModelT, gangs: Dict[str, Gang], card: AssetCard) -> None:
        raise RuntimeError(
            f"{card.name} has a sharded checkpoint, but has no model sharder. Please file a bug report."
        )


@final
class DelegatingModelLoader(ModelLoader[ModelT]):
    """Loads models of type ``ModelT`` using registered loaders."""

    _asset_store: AssetStore
    _loaders: Dict[str, ModelLoader[ModelT]]

    def __init__(self, *, asset_store: Optional[AssetStore] = None) -> None:
        """
        :param asset_store:
            The asset store where to check for available models.
        """
        self._asset_store = asset_store or default_asset_store

        self._loaders = {}

    def __call__(
        self,
        model_name_or_card: Union[str, AssetCard],
        *,
        gangs: Optional[Dict[str, Gang]] = None,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
        force: bool = False,
        progress: bool = True,
    ) -> ModelT:
        if isinstance(model_name_or_card, AssetCard):
            card = model_name_or_card
        else:
            card = self._asset_store.retrieve_card(model_name_or_card)

        family = card.field("model_family").as_(str)

        try:
            loader = self._loaders[family]
        except KeyError:
            raise AssetError(
                f"The value of the field 'model_family' of the asset card '{card.name}' must be a supported model family, but '{family}' has no registered loader."
            )

        return loader(
            model_name_or_card,
            gangs=gangs,
            device=device,
            dtype=dtype,
            force=force,
            progress=progress,
        )

    def register(self, family: str, loader: ModelLoader[ModelT]) -> None:
        """Register a model loader to use with this loader.

        :param family:
            The model family. If the 'model_family' field of an asset card
            matches this value, the specified ``loader`` will be used.
        :param loader:
            The model loader.
        """
        if family in self._loaders:
            raise ValueError(
                f"`family` must be a unique model family name, but '{family}' has already a registered loader."
            )

        self._loaders[family] = loader


load_model = DelegatingModelLoader[Module]()
