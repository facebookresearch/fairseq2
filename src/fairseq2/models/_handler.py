# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from functools import partial
from pathlib import Path
from typing import Any, Protocol, TypeVar, final

from torch import Tensor
from torch.nn import Module
from typing_extensions import override

try:
    from transformers import PretrainedConfig  # type: ignore[import-not-found]
except ImportError:
    raise ImportError("transformers is required for model export config handling")

from fairseq2.assets import (
    AssetCard,
    AssetCardError,
    AssetCardFieldNotFoundError,
    AssetDownloadManager,
)
from fairseq2.config_registry import ConfigNotFoundError, ConfigProvider
from fairseq2.data_type import DataType, default_dtype
from fairseq2.device import CPU, META_DEVICE
from fairseq2.error import ContractError, NotSupportedError
from fairseq2.gang import Gangs
from fairseq2.models.utils.checkpoint import load_checkpoint
from fairseq2.models.utils.sharder import ModelSharder, ShardSpec
from fairseq2.nn.data_parallel import FSDPGranularity, FSDPWrapper, load_with_sdp_gang
from fairseq2.nn.utils.module import (
    reset_non_persistent_buffers,
    to_device,
    to_empty,
)
from fairseq2.utils.merge import MergeError, merge_object
from fairseq2.utils.progress import ProgressReporter
from fairseq2.utils.structured import StructureError, structure, unstructure
from fairseq2.utils.validation import validate

# isort: split

from fairseq2.models._checkpoint import CheckpointError, CheckpointLoader
from fairseq2.models._error import (
    ModelConfigLoadError,
    ModelLoadError,
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
        mmap: bool = False,
        restrict: bool | None = None,
    ) -> Module: ...

    @abstractmethod
    def iter_checkpoint(
        self,
        path: Path,
        config: object,
        gangs: Gangs,
        *,
        mmap: bool = False,
        restrict: bool | None = None,
    ) -> Iterable[tuple[str, Tensor]]: ...

    @abstractmethod
    def compile(self, model: Module, *args: Any, **kwargs: Any) -> None: ...

    @abstractmethod
    def apply_activation_checkpointing(
        self, model: Module, *, every_nth_layer: int = 1
    ) -> None: ...

    @abstractmethod
    def apply_fsdp(
        self, model: Module, granularity: FSDPGranularity, wrapper: FSDPWrapper
    ) -> None: ...

    @abstractmethod
    def save_as_hugging_face(
        self, save_dir: Path, checkpoint: dict[str, object], config: object
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

    @property
    @abstractmethod
    def supports_hugging_face(self) -> bool: ...


ModelT_co = TypeVar("ModelT_co", bound=Module, covariant=True)

ModelConfigT_contra = TypeVar("ModelConfigT_contra", contravariant=True)


class ModelFactory(Protocol[ModelConfigT_contra, ModelT_co]):
    def __call__(self, config: ModelConfigT_contra) -> ModelT_co: ...


class CheckpointConverter(Protocol[ModelConfigT_contra]):
    def __call__(
        self, checkpoint: dict[str, object], config: ModelConfigT_contra
    ) -> dict[str, object]: ...


ModelT_contra = TypeVar("ModelT_contra", bound=Module, contravariant=True)


class ShardSpecsProvider(Protocol[ModelConfigT_contra]):
    def __call__(self, config: ModelConfigT_contra) -> dict[str, ShardSpec]: ...


class ModelCompiler(Protocol[ModelT_contra]):
    def __call__(self, model: ModelT_contra, *args: Any, **kwargs: Any) -> None: ...


class ActivationCheckpointApplier(Protocol[ModelT_contra]):
    def __call__(self, model: ModelT_contra, *, every_nth_layer: int = 1) -> None: ...


class FSDPApplier(Protocol[ModelT_contra]):
    def __call__(
        self, model: ModelT_contra, granularity: FSDPGranularity, wrapper: FSDPWrapper
    ) -> None: ...


class HuggingFaceSaver(Protocol[ModelConfigT_contra]):
    def __call__(
        self, save_dir: Path, checkpoint: dict[str, object], config: ModelConfigT_contra
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
    _checkpoint_loader: CheckpointLoader
    _sharder: ModelSharder
    _progress_reporter: ProgressReporter
    _supports_meta: bool
    _restrict: bool
    _checkpoint_converter: CheckpointConverter[Any] | None
    _shard_specs: ShardSpecsProvider[Any] | None
    _compiler: ModelCompiler[Any] | None
    _ac_applier: ActivationCheckpointApplier[Any] | None
    _fsdp_applier: FSDPApplier[Any] | None
    _hugging_face_saver: HuggingFaceSaver[Any] | None

    def __init__(
        self,
        family: str,
        kls: type[ModelT],
        configs: ConfigProvider[ModelConfigT],
        default_arch: str,
        factory: ModelFactory[ModelConfigT, ModelT],
        asset_download_manager: AssetDownloadManager,
        checkpoint_loader: CheckpointLoader,
        sharder: ModelSharder,
        progress_reporter: ProgressReporter,
        *,
        supports_meta: bool = True,
        restrict: bool = True,
        checkpoint_converter: CheckpointConverter[ModelConfigT] | None = None,
        shard_specs: ShardSpecsProvider[ModelConfigT] | None = None,
        compiler: ModelCompiler[ModelT] | None = None,
        ac_applier: ActivationCheckpointApplier[ModelT] | None = None,
        fsdp_applier: FSDPApplier[ModelT] | None = None,
        hugging_face_saver: HuggingFaceSaver[ModelConfigT] | None = None,
    ) -> None:
        self._family = family
        self._kls = kls
        self._configs = configs
        self._default_arch = default_arch
        self._factory = factory
        self._asset_download_manager = asset_download_manager
        self._checkpoint_loader = checkpoint_loader
        self._sharder = sharder
        self._progress_reporter = progress_reporter
        self._supports_meta = supports_meta
        self._restrict = restrict
        self._checkpoint_converter = checkpoint_converter
        self._shard_specs = shard_specs
        self._compiler = compiler
        self._ac_applier = ac_applier
        self._fsdp_applier = fsdp_applier
        self._hugging_face_saver = hugging_face_saver

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

        # Load the checkpoint.
        try:
            checkpoint_uri = card.field("checkpoint").as_uri()
        except AssetCardError as ex:
            raise model_asset_card_error(name) from ex

        path = self._asset_download_manager.download_checkpoint(checkpoint_uri, name)

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
                path, name, config, gangs, dtype, mmap=mmap, restrict=restrict
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
        mmap: bool = False,
        restrict: bool | None = None,
    ) -> Module:
        config = structure(config, self._configs.config_kls)

        validate(config)

        model = self._do_create(config, gangs, dtype, meta=self._supports_meta)

        if self._supports_meta:
            # The parameters of the model will be overwritten by the checkpoint,
            # so there is no need to redundantly initialize them.
            to_empty(model, device=gangs.root.device)

        checkpoint = self._do_iter_checkpoint(
            path, config, gangs, mmap=mmap, restrict=restrict
        )

        try:
            load_checkpoint(model, checkpoint, self._progress_reporter)
        except (CheckpointError, KeyError, ValueError) as ex:
            raise ModelLoadError(
                name, f"The checkpoint of the '{name}' model cannot be loaded. See the nested exception for details."  # fmt: skip
            ) from ex

        if self._supports_meta:
            # Non-persistent buffers are not included in the checkpoint, so we
            # have to explicitly initialize them.
            reset_non_persistent_buffers(model)

        return model

    @override
    def iter_checkpoint(
        self,
        path: Path,
        config: object,
        gangs: Gangs,
        *,
        mmap: bool = False,
        restrict: bool | None = None,
    ) -> Iterable[tuple[str, Tensor]]:
        config = structure(config, self._configs.config_kls)

        validate(config)

        return self._do_iter_checkpoint(
            path, config, gangs, mmap=mmap, restrict=restrict
        )

    def _do_iter_checkpoint(
        self,
        path: Path,
        config: object,
        gangs: Gangs,
        *,
        mmap: bool = False,
        restrict: bool | None = None,
    ) -> Iterable[tuple[str, Tensor]]:
        if gangs.root.device.type == "meta":
            raise ValueError(
                "`gangs.root` must be on a real device, but is on the meta device instead."
            )

        if restrict is None:
            restrict = self._restrict

        if self._checkpoint_converter is None:
            checkpoint_processor = None
        else:
            checkpoint_processor = partial(self._checkpoint_converter, config=config)

        if self._shard_specs is None:
            shard_specs = None
        else:
            shard_specs = self._shard_specs(config)

        with load_with_sdp_gang(gangs):  # Required for ShardedTensor
            yield from self._checkpoint_loader.load(
                path,
                gangs,
                mmap=mmap,
                restrict=restrict,
                processor=checkpoint_processor,
                shard_specs=shard_specs,
            )

    def _do_create(
        self, config: object, gangs: Gangs, dtype: DataType, meta: bool
    ) -> Module:
        if meta:
            if not self._supports_meta:
                raise NotSupportedError(
                    f"The '{self._family}' model family does not support meta device initialization."
                )

            device = META_DEVICE
        elif gangs.root.size != gangs.dp.size:
            device = CPU  # Avoid OOM for sharded models.
        else:
            device = gangs.root.device

        try:
            with device, default_dtype(dtype):
                model = self._factory(config)
        except NotImplementedError as ex:
            if "'Meta' backend" not in str(ex):
                raise

            raise ContractError(
                "One or more operators in the model constructor have failed to initialize on the meta device. See the nested exception for details."
            ) from ex

        if gangs.root.size != gangs.dp.size:
            if self._shard_specs is None:
                raise NotSupportedError(
                    f"The '{self._family}' model family does not support model parallelism."
                )

            shard_specs = self._shard_specs(config)

            try:
                self._sharder.shard(model, gangs, shard_specs)
            except ValueError as ex:
                raise ContractError(
                    "The model cannot be sharded. See the nested exception for details."
                ) from ex

            if not meta and device != gangs.root.device:
                to_device(model, gangs.root.device)

        return model

    @override
    def compile(self, model: Module, *args: Any, **kwargs: Any) -> None:
        if self._compiler is None:
            raise NotSupportedError(
                f"The '{self._family}' model family does not support `torch.compile()`."
            )

        if not isinstance(model, self._kls):
            raise TypeError(
                f"`model` must be of type `{self._kls}`, but is of type `{type(model)}` instead."
            )

        self._compiler(model, *args, **kwargs)

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
        self, model: Module, granularity: FSDPGranularity, wrapper: FSDPWrapper
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

    @override
    def save_as_hugging_face(
        self, save_dir: Path, checkpoint: dict[str, object], config: object
    ) -> None:
        if self._hugging_face_saver is None:
            raise NotSupportedError(
                f"The '{self._family}' model family does not support Hugging Face conversion."
            )

        if not isinstance(config, self._configs.config_kls):
            raise TypeError(
                f"`config` must be of type `{self._configs.config_kls}`, but is of type `{type(config)}` instead."
            )

        return self._hugging_face_saver(save_dir, checkpoint, config)

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
        return self._shard_specs is not None

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

    @property
    @override
    def supports_hugging_face(self) -> bool:
        return self._hugging_face_saver is not None
