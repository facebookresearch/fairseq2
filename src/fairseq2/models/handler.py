# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from errno import ENOENT
from functools import partial
from os import strerror
from pathlib import Path
from typing import Any, Protocol, TypeVar, final

import torch
from torch import Tensor
from torch.nn import Module
from typing_extensions import override

from fairseq2.assets import (
    AssetCard,
    AssetCardError,
    AssetConfigLoader,
    AssetDownloadManager,
)
from fairseq2.data_type import DataType, default_dtype
from fairseq2.device import CPU, META_DEVICE
from fairseq2.error import (
    InternalError,
    NotSupportedError,
    raise_operational_system_error,
)
from fairseq2.file_system import FileSystem
from fairseq2.gang import Gangs
from fairseq2.model_checkpoint import ModelCheckpointError, ModelCheckpointLoader
from fairseq2.models.utils.ac import apply_layerwise_activation_checkpointing
from fairseq2.models.utils.checkpoint import set_model_state
from fairseq2.models.utils.data_parallel import apply_layerwise_fsdp
from fairseq2.nn.data_parallel import FSDPGranularity, FSDPWrapper, load_with_sdp_gang
from fairseq2.nn.utils.module import reset_non_persistent_buffers, to_device, to_empty
from fairseq2.runtime.config_registry import (
    ConfigNotFoundError,
    ConfigProvider,
    StandardConfigProvider,
)
from fairseq2.runtime.dependency import DependencyContainer, DependencyResolver
from fairseq2.sharder import ModelSharder, ShardSpec
from fairseq2.utils.progress import NoopProgressReporter, ProgressReporter


class ModelFamilyHandler(ABC):
    @abstractmethod
    def get_archs(self) -> set[str]: ...

    @abstractmethod
    def get_arch_config(self, arch: str) -> object: ...

    @abstractmethod
    def get_model_config(self, card: AssetCard) -> object: ...

    @abstractmethod
    def create_new_model(
        self, config: object, gangs: Gangs, dtype: DataType, meta: bool
    ) -> Module: ...

    @abstractmethod
    def load_model(
        self,
        card: AssetCard,
        gangs: Gangs,
        dtype: DataType,
        config: object | None,
        mmap: bool,
        progress: bool,
    ) -> Module: ...

    @abstractmethod
    def load_custom_model(
        self,
        path: Path,
        config: object,
        gangs: Gangs,
        dtype: DataType,
        mmap: bool,
        restrict: bool | None,
        progress: bool,
    ) -> Module: ...

    @abstractmethod
    def iter_checkpoint(
        self,
        path: Path,
        config: object,
        gangs: Gangs,
        mmap: bool,
        restrict: bool | None,
    ) -> Iterable[tuple[str, Tensor]]: ...

    @abstractmethod
    def compile(self, model: Module, *args: Any, **kwargs: Any) -> None: ...

    @abstractmethod
    def apply_activation_checkpointing(
        self, model: Module, every_nth_layer: int
    ) -> None: ...

    @abstractmethod
    def apply_fsdp(
        self, model: Module, granularity: FSDPGranularity, wrapper: FSDPWrapper
    ) -> None: ...

    @abstractmethod
    def save_as_hugging_face(
        self, save_dir: Path, state_dict: dict[str, object], config: object
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


class StateDictConverter(Protocol[ModelConfigT_contra]):
    def __call__(
        self, state_dict: dict[str, object], config: ModelConfigT_contra
    ) -> dict[str, object]: ...


class ShardSpecsProvider(Protocol[ModelConfigT_contra]):
    def __call__(self, config: ModelConfigT_contra) -> dict[str, ShardSpec]: ...


ModelT_contra = TypeVar("ModelT_contra", bound=Module, contravariant=True)


class TorchCompiler(Protocol[ModelT_contra]):
    def __call__(self, model: ModelT_contra, *args: Any, **kwargs: Any) -> None: ...


class ActivationCheckpointApplier(Protocol[ModelT_contra]):
    def __call__(self, model: ModelT_contra, *, every_nth_layer: int) -> None: ...


class FSDPApplier(Protocol[ModelT_contra]):
    def __call__(
        self, model: ModelT_contra, granularity: FSDPGranularity, wrapper: FSDPWrapper
    ) -> None: ...


class HuggingFaceSaver(Protocol[ModelConfigT_contra]):
    def __call__(
        self, save_dir: Path, state_dict: dict[str, object], config: ModelConfigT_contra
    ) -> None: ...


ModelT = TypeVar("ModelT", bound=Module)

ModelConfigT = TypeVar("ModelConfigT")


@final
class StandardModelFamilyHandler(ModelFamilyHandler):
    _kls: type[Module]
    _configs: ConfigProvider[object]
    _factory: ModelFactory[Any, Module]
    _state_dict_converter: StateDictConverter[Any] | None
    _shard_specs: ShardSpecsProvider[Any] | None
    _compiler: TorchCompiler[Any] | None
    _ac_applier: ActivationCheckpointApplier[Any] | None
    _fsdp_applier: FSDPApplier[Any] | None
    _hugging_face_saver: HuggingFaceSaver[Any] | None

    def __init__(
        self,
        family: str,
        kls: type[ModelT],
        configs: ConfigProvider[ModelConfigT],
        factory: ModelFactory[ModelConfigT, ModelT],
        file_system: FileSystem,
        asset_download_manager: AssetDownloadManager,
        asset_config_loader: AssetConfigLoader,
        checkpoint_loader: ModelCheckpointLoader,
        sharder: ModelSharder,
        progress_reporter: ProgressReporter,
        *,
        supports_meta: bool = True,
        restrict: bool = True,
        state_dict_converter: StateDictConverter[ModelConfigT] | None = None,
        shard_specs: ShardSpecsProvider[ModelConfigT] | None = None,
        compiler: TorchCompiler[ModelT] | None = None,
        ac_applier: ActivationCheckpointApplier[ModelT] | None = None,
        fsdp_applier: FSDPApplier[ModelT] | None = None,
        hugging_face_saver: HuggingFaceSaver[ModelConfigT] | None = None,
    ) -> None:
        self._family = family
        self._kls = kls
        self._configs = configs
        self._factory = factory
        self._file_system = file_system
        self._asset_download_manager = asset_download_manager
        self._asset_config_loader = asset_config_loader
        self._checkpoint_loader = checkpoint_loader
        self._sharder = sharder
        self._progress_reporter = progress_reporter
        self._supports_meta = supports_meta
        self._restrict = restrict
        self._state_dict_converter = state_dict_converter
        self._shard_specs = shard_specs
        self._compiler = compiler
        self._ac_applier = ac_applier
        self._fsdp_applier = fsdp_applier
        self._hugging_face_saver = hugging_face_saver

    @override
    def get_archs(self) -> set[str]:
        it = self._configs.get_config_names()

        return set(it)

    @override
    def get_arch_config(self, arch: str) -> object:
        return self._configs.get_config(arch)

    @override
    def get_model_config(self, card: AssetCard) -> object:
        name = card.name

        arch_field = card.maybe_get_field("model_arch")
        if arch_field is not None:
            arch = arch_field.as_(str)
        else:
            arch = None

        if arch is None:
            try:
                base_config = self._configs.kls()
            except TypeError as ex:
                raise InternalError(
                    f"Default configuration of the {self._family} model family cannot be constructed."
                ) from ex
        else:
            try:
                base_config = self.get_arch_config(arch)
            except ConfigNotFoundError:
                msg = f"model_arch field of the {name} asset card is expected to be a supported model architecture, but is {arch} instead."

                raise AssetCardError(name, msg) from None

        return self._asset_config_loader.load(
            card, base_config, config_key="model_config"
        )

    @override
    def create_new_model(
        self, config: object, gangs: Gangs, dtype: DataType, meta: bool
    ) -> Module:
        if not isinstance(config, self._configs.kls):
            raise TypeError(
                f"`config` must be of type `{self._configs.kls}`, but is of type `{type(config)}` instead."
            )

        return self._do_create_model(config, gangs, dtype, meta)

    @override
    def load_model(
        self,
        card: AssetCard,
        gangs: Gangs,
        dtype: DataType,
        config: object | None,
        mmap: bool,
        progress: bool,
    ) -> Module:
        name = card.name

        uri = card.field("checkpoint").as_uri()

        if uri.scheme not in self._asset_download_manager.supported_schemes:
            msg = f"checkpoint URI scheme of the {name} asset card is expected to be a supported scheme, but is {uri.scheme} instead."

            raise AssetCardError(name, msg)

        path = self._asset_download_manager.download_model(uri, name, progress=progress)

        # Handle legacy paths with format specifiers.
        if "shard_idx" in path.name:
            path = path.parent

        # Load the configuration.
        if config is None:
            config = self.get_model_config(card)

            has_custom_config = False
        else:
            if not isinstance(config, self._configs.kls):
                raise TypeError(
                    f"`config` must be of type `{self._configs.kls}`, but is of type `{type(config)}` instead."
                )

            has_custom_config = True

        restrict_field = card.maybe_get_field("restrict")
        if restrict_field is not None:
            restrict = restrict_field.as_(bool)
        else:
            restrict = self._restrict

        try:
            return self._do_load_model(
                path, config, gangs, dtype, mmap, restrict, progress
            )
        except ValueError as ex:
            if has_custom_config:
                raise

            msg = f"model_config field of the {name} asset card is not a valid {self._family} model configuration."

            raise AssetCardError(name, msg) from ex
        except ModelCheckpointError as ex:
            msg = f"Model checkpoint of the {name} asset card cannot be loaded."

            raise AssetCardError(name, msg) from ex
        except FileNotFoundError as ex:
            if uri.scheme != "file":
                raise_operational_system_error(ex)

            msg = f"{path} pointed to by the checkpoint field of the {name} asset card is not found."

            raise AssetCardError(name, msg)
        except OSError as ex:
            raise_operational_system_error(ex)

    @override
    def load_custom_model(
        self,
        path: Path,
        config: object,
        gangs: Gangs,
        dtype: DataType,
        mmap: bool,
        restrict: bool | None,
        progress: bool,
    ) -> Module:
        if not isinstance(config, self._configs.kls):
            raise TypeError(
                f"`config` must be of type `{self._configs.kls}`, but is of type `{type(config)}` instead."
            )

        return self._do_load_model(path, config, gangs, dtype, mmap, restrict, progress)

    def _do_load_model(
        self,
        path: Path,
        config: object,
        gangs: Gangs,
        dtype: DataType,
        mmap: bool,
        restrict: bool | None,
        progress: bool,
    ) -> Module:
        path_exists = self._file_system.exists(path)
        if not path_exists:
            raise FileNotFoundError(ENOENT, strerror(ENOENT), path)

        model = self._do_create_model(config, gangs, dtype, self._supports_meta)

        if self._supports_meta:
            # The parameters of the model will be overwritten by the checkpoint,
            # so there is no need to redundantly initialize them.
            to_empty(model, device=gangs.root.device)

        checkpoint = self._do_iter_checkpoint(path, config, gangs, mmap, restrict)

        pr = self._progress_reporter if progress else NoopProgressReporter()

        try:
            set_model_state(model, checkpoint, pr)
        except ValueError as ex:
            msg = (
                f"Checkpoint at {path} is not compatible with the {self._family} model."
            )

            raise ModelCheckpointError(path, msg) from ex

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
        mmap: bool,
        restrict: bool | None,
    ) -> Iterable[tuple[str, Tensor]]:
        if not isinstance(config, self._configs.kls):
            raise TypeError(
                f"`config` must be of type `{self._configs.kls}`, but is of type `{type(config)}` instead."
            )

        return self._do_iter_checkpoint(path, config, gangs, mmap, restrict)

    def _do_iter_checkpoint(
        self,
        path: Path,
        config: object,
        gangs: Gangs,
        mmap: bool,
        restrict: bool | None,
    ) -> Iterable[tuple[str, Tensor]]:
        if gangs.root.device.type == "meta":
            raise ValueError(
                "`gangs.root` must be on a real device, but is on the meta device instead."
            )

        if restrict is None:
            restrict = self._restrict

        if self._state_dict_converter is None:
            state_dict_converter = None
        else:
            state_dict_converter = partial(self._state_dict_converter, config=config)

        if self._shard_specs is None:
            shard_specs = None
        else:
            shard_specs = self._shard_specs(config)

        with load_with_sdp_gang(gangs):  # Required for ShardedTensor
            yield from self._checkpoint_loader.lazy_load(
                path,
                gangs,
                mmap=mmap,
                restrict=restrict,
                state_dict_converter=state_dict_converter,
                shard_specs=shard_specs,
            )

    def _do_create_model(
        self, config: object, gangs: Gangs, dtype: DataType, meta: bool
    ) -> Module:
        if meta:
            if not self._supports_meta:
                raise NotSupportedError(
                    f"{self._family} model family does not support meta device initialization."
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

            raise InternalError(
                "One or more modules of the model failed to initialize on the meta device."
            ) from ex

        if gangs.root.size != gangs.dp.size:
            if self._shard_specs is None:
                raise NotSupportedError(
                    f"{self._family} model family does not support model parallelism."
                )

            shard_specs = self._shard_specs(config)

            try:
                self._sharder.shard(model, gangs, shard_specs)
            except ValueError as ex:
                raise InternalError("Model cannot be sharded.") from ex

            if not meta and device != gangs.root.device:
                to_device(model, gangs.root.device)

        return model

    @override
    def compile(self, model: Module, *args: Any, **kwargs: Any) -> None:
        if self._compiler is None:
            raise NotSupportedError(
                f"{self._family} model family does not support torch.compile()."
            )

        if not isinstance(model, self._kls):
            raise TypeError(
                f"`model` must be of type `{self._kls}`, but is of type `{type(model)}` instead."
            )

        self._compiler(model, *args, **kwargs)

    @override
    def apply_activation_checkpointing(
        self, model: Module, every_nth_layer: int
    ) -> None:
        if self._ac_applier is None:
            raise NotSupportedError(
                f"{self._family} model family does not support activation checkpointing."
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
                f"{self._family} model family does not support FSDP."
            )

        if not isinstance(model, self._kls):
            raise TypeError(
                f"`model` must be of type `{self._kls}`, but is of type `{type(model)}` instead."
            )

        self._fsdp_applier(model, granularity, wrapper)

    @override
    def save_as_hugging_face(
        self, save_dir: Path, state_dict: dict[str, object], config: object
    ) -> None:
        if self._hugging_face_saver is None:
            raise NotSupportedError(
                f"{self._family} model family does not support Hugging Face conversion."
            )

        if not isinstance(config, self._configs.kls):
            raise TypeError(
                f"`config` must be of type `{self._configs.kls}`, but is of type `{type(config)}` instead."
            )

        return self._hugging_face_saver(save_dir, state_dict, config)

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
        return self._configs.kls

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


class AdvancedModelFactory(Protocol[ModelConfigT_contra, ModelT_co]):
    def __call__(
        self, resolver: DependencyResolver, config: ModelConfigT_contra
    ) -> ModelT_co: ...


def register_model_family(
    container: DependencyContainer,
    family: str,
    kls: type[ModelT],
    config_kls: type[ModelConfigT],
    *,
    factory: ModelFactory[ModelConfigT, ModelT] | None = None,
    advanced_factory: AdvancedModelFactory[ModelConfigT, ModelT] | None = None,
    supports_meta: bool = True,
    supports_compilation: bool = True,
    supports_ac: bool = True,
    supports_fsdp: bool = True,
    restrict: bool = True,
    state_dict_converter: StateDictConverter[ModelConfigT] | None = None,
    shard_specs: ShardSpecsProvider[ModelConfigT] | None = None,
    compiler: TorchCompiler[ModelT] | None = None,
    ac_applier: ActivationCheckpointApplier[ModelT] | None = None,
    fsdp_applier: FSDPApplier[ModelT] | None = None,
    hugging_face_saver: HuggingFaceSaver[ModelConfigT] | None = None,
) -> None:
    def create_handler(resolver: DependencyResolver) -> ModelFamilyHandler:
        nonlocal factory

        if advanced_factory is not None:
            if factory is not None:
                raise ValueError(
                    "`factory` and `advanced_factory` must not be specified at the same time."
                )

            def create_model(config: ModelConfigT) -> ModelT:
                return advanced_factory(resolver, config)

            factory = create_model
        elif factory is None:
            raise ValueError("`factory` or `advanced_factory` must be specified.")

        file_system = resolver.resolve(FileSystem)

        asset_download_manager = resolver.resolve(AssetDownloadManager)

        checkpoint_loader = resolver.resolve(ModelCheckpointLoader)

        sharder = resolver.resolve(ModelSharder)

        progress_reporter = resolver.resolve(ProgressReporter)

        asset_config_loader = resolver.resolve(AssetConfigLoader)

        configs = StandardConfigProvider(resolver, config_kls)

        nonlocal compiler

        if supports_compilation:
            if compiler is None:

                def compile(model: ModelT, **kwargs: Any) -> None:
                    torch.compile(model)

                compiler = compile
        else:
            compiler = None

        nonlocal ac_applier

        if supports_ac:
            if ac_applier is None:
                ac_applier = apply_layerwise_activation_checkpointing
        else:
            ac_applier = None

        nonlocal fsdp_applier

        if supports_fsdp:
            if fsdp_applier is None:
                fsdp_applier = apply_layerwise_fsdp
        else:
            fsdp_applier = None

        return StandardModelFamilyHandler(
            family,
            kls,
            configs,
            factory,
            file_system,
            asset_download_manager,
            asset_config_loader,
            checkpoint_loader,
            sharder,
            progress_reporter,
            supports_meta=supports_meta,
            restrict=restrict,
            state_dict_converter=state_dict_converter,
            shard_specs=shard_specs,
            compiler=compiler,
            ac_applier=ac_applier,
            fsdp_applier=fsdp_applier,
            hugging_face_saver=hugging_face_saver,
        )

    container.register(ModelFamilyHandler, create_handler, key=family)
