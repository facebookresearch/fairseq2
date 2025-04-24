# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Protocol, TypeVar, final

from torch.nn import Module
from typing_extensions import override

from fairseq2.assets import (
    AssetCard,
    AssetCardError,
    AssetCardFieldNotFoundError,
    AssetDownloadManager,
)
from fairseq2.config_registry import ConfigNotFoundError, ConfigProvider
from fairseq2.device import default_device_and_dtype
from fairseq2.error import ContractError, NotSupportedError
from fairseq2.gang import Gangs
from fairseq2.nn.data_parallel import FsdpGranularity, FsdpWrapper, load_with_sdp_gang
from fairseq2.nn.utils.module import (
    load_state_dict,
    reset_non_persistent_buffers,
    to_device,
    to_empty,
)
from fairseq2.typing import CPU, META, DataType
from fairseq2.utils.io import TensorLoader, TensorLoadError
from fairseq2.utils.merge import MergeError, merge_object
from fairseq2.utils.structured import StructureError, structure, unstructure
from fairseq2.utils.validation import validate

# isort: split

from fairseq2.models._error import (
    ModelConfigLoadError,
    ModelLoadError,
    ShardedModelLoadError,
    UnknownModelArchitectureError,
    model_asset_card_error,
)


class ModelHandler(ABC):
    @abstractmethod
    def get_arch_config(self, arch: str | None) -> object: ...

    @abstractmethod
    def load_config(self, card: AssetCard) -> object: ...

    @abstractmethod
    def create(
        self, config: object, gangs: Gangs, dtype: DataType, meta: bool
    ) -> Module: ...

    @abstractmethod
    def load(
        self,
        card: AssetCard,
        gangs: Gangs,
        dtype: DataType,
        config: object,
        *,
        mmap: bool = False,
    ) -> Module: ...

    @abstractmethod
    def load_from_path(
        self,
        path: Path,
        name: str,
        config: object,
        gangs: Gangs,
        dtype: DataType,
        *,
        restrict: bool | None = None,
        mmap: bool = False,
    ) -> Module: ...

    @abstractmethod
    def compile(self, model: Module, **kwargs: Any) -> None: ...

    @abstractmethod
    def apply_activation_checkpointing(
        self, model: Module, *, every_nth_layer: int = 1
    ) -> None: ...

    @abstractmethod
    def apply_fsdp(
        self, model: Module, granularity: FsdpGranularity, wrapper: FsdpWrapper
    ) -> None: ...

    @property
    @abstractmethod
    def family(self) -> str: ...

    @property
    @abstractmethod
    def kls(self) -> type[Module]: ...

    @property
    @abstractmethod
    def config_kls(self) -> type[object]: ...

    @property
    @abstractmethod
    def supports_meta(self) -> bool: ...

    @property
    @abstractmethod
    def supports_model_parallelism(self) -> bool: ...

    @property
    @abstractmethod
    def supports_compilation(self) -> bool: ...

    @property
    @abstractmethod
    def supports_activation_checkpointing(self) -> bool: ...

    @property
    @abstractmethod
    def supports_fsdp(self) -> bool: ...


ModelT_co = TypeVar("ModelT_co", bound=Module, covariant=True)

ModelConfigT_contra = TypeVar("ModelConfigT_contra", contravariant=True)


class ModelFactory(Protocol[ModelConfigT_contra, ModelT_co]):
    def __call__(self, config: ModelConfigT_contra) -> ModelT_co: ...


class CheckpointConverter(Protocol[ModelConfigT_contra]):
    def __call__(
        self, checkpoint: dict[str, object], config: ModelConfigT_contra
    ) -> dict[str, object]: ...


ModelT_contra = TypeVar("ModelT_contra", bound=Module, contravariant=True)


class ModelSharder(Protocol[ModelT_contra, ModelConfigT_contra]):
    def __call__(
        self, model: ModelT_contra, config: ModelConfigT_contra, gangs: Gangs
    ) -> None: ...


class ModelCompiler(Protocol[ModelT_contra]):
    def __call__(self, model: ModelT_contra, **kwargs: Any) -> None: ...


class ActivationCheckpointApplier(Protocol[ModelT_contra]):
    def __call__(self, model: ModelT_contra, *, every_nth_layer: int = 1) -> None: ...


class FsdpApplier(Protocol[ModelT_contra]):
    def __call__(
        self, model: ModelT_contra, granularity: FsdpGranularity, wrapper: FsdpWrapper
    ) -> None: ...


ModelT = TypeVar("ModelT", bound=Module)

ModelConfigT = TypeVar("ModelConfigT")


@final
class DelegatingModelHandler(ModelHandler):
    _family: str
    _kls: type[Module]
    _configs: ConfigProvider[object]
    _default_arch: str
    _factory: ModelFactory[Any, Module]
    _asset_download_manager: AssetDownloadManager
    _tensor_loader: TensorLoader
    _supports_meta: bool
    _restrict: bool
    _checkpoint_converter: CheckpointConverter[Any] | None
    _sharder: ModelSharder[Any, Any] | None
    _compiler: ModelCompiler[Any] | None
    _ac_applier: ActivationCheckpointApplier[Any] | None
    _fsdp_applier: FsdpApplier[Any] | None

    def __init__(
        self,
        family: str,
        kls: type[ModelT],
        configs: ConfigProvider[ModelConfigT],
        default_arch: str,
        factory: ModelFactory[ModelConfigT, ModelT],
        asset_download_manager: AssetDownloadManager,
        tensor_loader: TensorLoader,
        *,
        supports_meta: bool = True,
        restrict: bool = True,
        checkpoint_converter: CheckpointConverter[ModelConfigT] | None = None,
        sharder: ModelSharder[ModelT, ModelConfigT] | None = None,
        compiler: ModelCompiler[ModelT] | None = None,
        ac_applier: ActivationCheckpointApplier[ModelT] | None = None,
        fsdp_applier: FsdpApplier[ModelT] | None = None,
    ) -> None:
        self._family = family
        self._kls = kls
        self._configs = configs
        self._default_arch = default_arch
        self._factory = factory
        self._asset_download_manager = asset_download_manager
        self._tensor_loader = tensor_loader
        self._supports_meta = supports_meta
        self._restrict = restrict
        self._checkpoint_converter = checkpoint_converter
        self._sharder = sharder
        self._compiler = compiler
        self._ac_applier = ac_applier
        self._fsdp_applier = fsdp_applier

    @override
    def get_arch_config(self, arch: str | None) -> object:
        if arch is None:
            effective_arch = self._default_arch
        else:
            effective_arch = arch

        try:
            return self._configs.get(effective_arch)
        except ConfigNotFoundError:
            if arch is None:
                raise ContractError(
                    f"The '{self._family}' model family does not have a configuration for the default '{self._default_arch}' architecture."
                )

            raise

    @override
    def load_config(self, card: AssetCard) -> object:
        name = card.name

        try:
            arch = card.field("model_arch").as_(str)
        except AssetCardFieldNotFoundError:
            arch = None
        except AssetCardError as ex:
            raise ModelConfigLoadError(
                name, f"The '{name}' asset card cannot be read. See the nested exception for details."  # fmt: skip
            ) from ex

        try:
            config = self.get_arch_config(arch)
        except ConfigNotFoundError:
            if arch is not None:
                raise UnknownModelArchitectureError(arch, self._family, name) from None

            raise

        # Override the default architecture configuration if the asset card or
        # its bases have a 'model_config' field.
        config_overrides = []

        base_card: AssetCard | None = card

        while base_card is not None:
            if "model_config" in base_card.metadata:
                config_override_field = base_card.field("model_config")

                config_override = config_override_field.as_unstructured()

                config_overrides.append(config_override)

            base_card = base_card.base

        if config_overrides:
            try:
                unstructured_config = unstructure(config)
            except StructureError as ex:
                raise ContractError(
                    f"The configuration class of the '{self._family}' model family cannot be unstructured. See the nested exception for details."
                ) from ex

            try:
                for config_override in reversed(config_overrides):
                    unstructured_config = merge_object(
                        unstructured_config, config_override
                    )

                config = structure(unstructured_config, type(config))
            except MergeError as ex:
                raise ModelConfigLoadError(
                    name, f"The '{name}' asset card does not contain a valid model configuration. See the nested exception for details."  # fmt: skip
                ) from ex

        return config

    @override
    def create(
        self, config: object, gangs: Gangs, dtype: DataType, meta: bool
    ) -> Module:
        config = structure(config, self._configs.config_kls)

        validate(config)

        return self._do_create(config, gangs, dtype, meta)

    @override
    def load(
        self,
        card: AssetCard,
        gangs: Gangs,
        dtype: DataType,
        config: object,
        *,
        mmap: bool = False,
    ) -> Module:
        name = card.name

        try:
            num_shards = card.field("num_shards").as_(int)
            if num_shards < 1:
                raise AssetCardError(
                    name, f"The value of the 'num_shards' field of the '{name}' asset card is expected to be a positive integer, but is {num_shards} instead."  # fmt: skip
                )
        except AssetCardFieldNotFoundError:
            num_shards = 1
        except AssetCardError as ex:
            raise model_asset_card_error(name) from ex

        if num_shards > 1 and gangs.tp.size != num_shards:
            raise ShardedModelLoadError(name, num_shards, gangs.tp.size)

        # Load the checkpoint.
        try:
            checkpoint_uri = card.field("checkpoint").as_uri()
        except AssetCardError as ex:
            raise model_asset_card_error(name) from ex

        shard_idx = gangs.tp.rank if num_shards > 1 else 0

        path = self._asset_download_manager.download_checkpoint(
            checkpoint_uri, name, shard_idx=shard_idx
        )

        # Load the configuration.
        if config is None:
            try:
                config = self.load_config(card)
            except ModelConfigLoadError as ex:
                raise ModelLoadError(
                    name, f"The '{name}' model configuration cannot be loaded. See the nested exception for details."  # fmt: skip
                ) from ex

            has_custom_config = False
        else:
            has_custom_config = True

        try:
            restrict = card.field("restrict").as_(bool)
        except AssetCardFieldNotFoundError:
            restrict = None
        except AssetCardError as ex:
            raise model_asset_card_error(name) from ex

        try:
            return self.load_from_path(
                path, name, config, gangs, dtype, restrict=restrict, mmap=mmap
            )
        except FileNotFoundError:
            raise ModelLoadError(
                name, f"The '{name}' model cannot be found at the '{path}' path."  # fmt: skip
            ) from None
        except ValueError as ex:
            if has_custom_config:
                raise

            raise ModelLoadError(
                name, f"The '{name}' asset card does not contain a valid model configuration of the '{self._family}' family. See the nested exception for details."  # fmt: skip
            ) from ex

    @override
    def load_from_path(
        self,
        path: Path,
        name: str,
        config: object,
        gangs: Gangs,
        dtype: DataType,
        *,
        restrict: bool | None = None,
        mmap: bool = False,
    ) -> Module:
        if gangs.root.device.type == "meta":
            raise ValueError(
                "`gangs` must be on a real device, but is on the meta device instead."
            )

        config = structure(config, self._configs.config_kls)

        validate(config)

        # Create the model.
        model = self._do_create(config, gangs, dtype, meta=self._supports_meta)

        if self._supports_meta:
            # Move the model to the actual device without initializing. Its
            # state will be overwritten by the checkpoint anyways.
            to_empty(model, device=gangs.root.device)

        if restrict is None:
            restrict = self._restrict

        with load_with_sdp_gang(gangs):  # Required for ShardedTensor
            try:
                checkpoint = self._tensor_loader.load(
                    path, map_location=CPU, restrict=restrict, mmap=mmap
                )
            except TensorLoadError as ex:
                raise ModelLoadError(
                    name, f"The checkpoint of the '{name}' model cannot be loaded. See the nested exception for details."  # fmt: skip
                ) from ex

        if "fs2" not in checkpoint:
            if self._checkpoint_converter is None:
                raise ModelLoadError(
                    name, f"The checkpoint of the '{name}' model is not fairseq2 compatible."  # fmt: skip
                )

            try:
                checkpoint = self._checkpoint_converter(checkpoint, config)
            except (KeyError, ValueError) as ex:
                raise ModelLoadError(
                    name, f"The checkpoint of the '{name}' model cannot be converted to a fairseq2 compatible format. See the nested exception for details."  # fmt: skip
                ) from ex

        # Load the model state.
        model_key = checkpoint.get("model_key", "model")

        if not isinstance(model_key, str):
            raise ModelLoadError(
                name, f"The 'model_key' in the '{name}' checkpoint is expected to be of type `str`, but is of type `{type(model_key)}` instead."  # fmt: skip
            )

        try:
            state_dict = checkpoint[model_key]
        except KeyError:
            raise ModelLoadError(
                name, f"The '{name}' checkpoint does not contain a '{model_key}' key."  # fmt: skip
            ) from None

        if not isinstance(state_dict, dict):
            raise ModelLoadError(
                name, f"The model state dictionary in the '{name}' checkpoint is expected to be of type `dict`, but is of type `{type(state_dict)}` instead."  # fmt: skip
            )

        try:
            load_state_dict(model, state_dict)
        except (KeyError, ValueError) as ex:
            raise ModelLoadError(
                name, f"The state of the '{name}' model cannot be loaded from the checkpoint. See the nested exception for details."  # fmt: skip
            ) from ex

        if self._supports_meta:
            # Non-persistent buffers are not included in the checkpoint, so we
            # have to explicitly initialize them.
            reset_non_persistent_buffers(model)

        return model

    def _do_create(
        self, config: object, gangs: Gangs, dtype: DataType, meta: bool
    ) -> Module:
        if meta:
            if not self._supports_meta:
                raise NotSupportedError(
                    f"The '{self._family}' model family does not support meta device initialization."
                )

            device = META
        elif gangs.root.size != gangs.dp.size:
            device = CPU  # Avoid OOM for sharded models.
        else:
            device = gangs.root.device

        try:
            with default_device_and_dtype(device, dtype):
                model = self._factory(config)
        except NotImplementedError as ex:
            if "'Meta' backend" not in str(ex):
                raise

            raise ContractError(
                "One or more operators in the model constructor have failed to initialize on the meta device. See the nested exception for details."
            ) from ex

        if gangs.root.size != gangs.dp.size:
            if self._sharder is None:
                raise NotSupportedError(
                    f"The '{self._family}' model family does not support model parallelism."
                )

            self._sharder(model, config, gangs)

            if not meta and device != gangs.root.device:
                to_device(model, gangs.root.device)

        return model

    @override
    def compile(self, model: Module, **kwargs: Any) -> None:
        if self._compiler is None:
            raise NotSupportedError(
                f"The '{self._family}' model family does not support `torch.compile()`."
            )

        if not isinstance(model, self._kls):
            raise TypeError(
                f"`model` must be of type `{self._kls}`, but is of type `{type(model)}` instead."
            )

        self._compiler(model, **kwargs)

    @override
    def apply_activation_checkpointing(
        self, model: Module, *, every_nth_layer: int = 1
    ) -> None:
        if self._ac_applier is None:
            raise NotSupportedError(
                f"The '{self._family}' model family does not support activation checkpointing."
            )

        if not isinstance(model, self._kls):
            raise TypeError(
                f"`model` must be of type `{self._kls}`, but is of type `{type(model)}` instead."
            )

        self._ac_applier(model, every_nth_layer=every_nth_layer)

    @override
    def apply_fsdp(
        self, model: Module, granularity: FsdpGranularity, wrapper: FsdpWrapper
    ) -> None:
        if self._fsdp_applier is None:
            raise NotSupportedError(
                f"The '{self._family}' model family does not support FSDP."
            )

        if not isinstance(model, self._kls):
            raise TypeError(
                f"`model` must be of type `{self._kls}`, but is of type `{type(model)}` instead."
            )

        self._fsdp_applier(model, granularity, wrapper)

    @property
    @override
    def family(self) -> str:
        return self._family

    @property
    @override
    def kls(self) -> type[Module]:
        return self._kls

    @property
    @override
    def config_kls(self) -> type[object]:
        return self._configs.config_kls

    @property
    @override
    def supports_meta(self) -> bool:
        return self._supports_meta

    @property
    @override
    def supports_model_parallelism(self) -> bool:
        return self._sharder is not None

    @property
    @override
    def supports_compilation(self) -> bool:
        return self._compiler is not None

    @property
    @override
    def supports_activation_checkpointing(self) -> bool:
        return self._ac_applier is not None

    @property
    @override
    def supports_fsdp(self) -> bool:
        return self._fsdp_applier is not None
