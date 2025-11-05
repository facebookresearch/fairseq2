# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
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
from fairseq2.device import META_DEVICE
from fairseq2.error import (
    InternalError,
    NotSupportedError,
    raise_operational_system_error,
)
from fairseq2.file_system import FileSystem, raise_if_not_exists
from fairseq2.gang import Gangs
from fairseq2.model_checkpoint import ModelCheckpointError, ModelCheckpointLoader
from fairseq2.models.utils.checkpoint import (
    ModelCheckpointMismatchError,
    set_model_state,
)
from fairseq2.nn import get_shard_dims
from fairseq2.nn.fsdp import FSDPWrapper, load_with_sdp_gang
from fairseq2.nn.utils.module import reset_non_persistent_buffers, to_empty
from fairseq2.runtime.lookup import Lookup
from fairseq2.sharder import ModelSharder, ShardSpec, ShardSpecError
from fairseq2.utils.progress import ProgressReporter
from fairseq2.utils.warn import _warn_deprecated


@dataclass
class HuggingFaceExport:
    state_dict: Mapping[str, object]
    config: Mapping[str, object]
    config_kls_name: str
    arch: str | Sequence[str]


class ModelFamily(ABC):
    @abstractmethod
    def get_archs(self) -> set[str]: ...

    @abstractmethod
    def maybe_get_arch_config(self, arch: str) -> object | None: ...

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
    ) -> Iterator[tuple[str, Tensor]]: ...

    @abstractmethod
    def compile(self, model: Module, *args: Any, **kwargs: Any) -> None: ...

    @abstractmethod
    def apply_fsdp(
        self, model: Module, granularity: str, wrapper: FSDPWrapper
    ) -> Module: ...

    @abstractmethod
    def apply_layerwise_ac(self, model: Module, every_nth_layer: int) -> Module: ...

    @abstractmethod
    def convert_to_hugging_face(
        self, state_dict: dict[str, object], config: object
    ) -> HuggingFaceExport: ...

    @property
    @abstractmethod
    def name(self) -> str: ...

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
    def supports_fsdp(self) -> bool: ...

    @property
    @abstractmethod
    def supports_layerwise_ac(self) -> bool: ...

    @property
    @abstractmethod
    def supports_hugging_face(self) -> bool: ...


def get_model_family(card: AssetCard, families: Lookup[ModelFamily]) -> ModelFamily:
    family_name = card.field("model_family").as_(str)

    family = families.maybe_get(family_name)
    if family is None:
        msg = f"family field of the {card.name} asset card is expected to be a supported model family, but is {family_name} instead."

        raise AssetCardError(card.name, msg)

    return family


ModelT_co = TypeVar("ModelT_co", bound=Module, covariant=True)

ModelConfigT_contra = TypeVar("ModelConfigT_contra", contravariant=True)


class ModelFactory(Protocol[ModelConfigT_contra, ModelT_co]):
    def __call__(self, config: ModelConfigT_contra) -> ModelT_co: ...


class ModelStateDictConverter(Protocol[ModelConfigT_contra]):
    def __call__(
        self, state_dict: dict[str, object], config: ModelConfigT_contra
    ) -> dict[str, object]: ...


class ShardSpecsProvider(Protocol[ModelConfigT_contra]):
    def __call__(self, config: ModelConfigT_contra) -> Mapping[str, ShardSpec]: ...


ModelT_contra = TypeVar("ModelT_contra", bound=Module, contravariant=True)


class ModelCompiler(Protocol[ModelT_contra]):
    def __call__(self, model: ModelT_contra, *args: Any, **kwargs: Any) -> None: ...


class ModelFSDPApplier(Protocol[ModelT_contra]):
    def __call__(
        self, model: ModelT_contra, granularity: str, wrapper: FSDPWrapper
    ) -> Module: ...


class LayerwiseACApplier(Protocol[ModelT_contra]):
    def __call__(self, model: ModelT_contra, every_nth_layer: int) -> Module: ...


class HuggingFaceExporter(Protocol[ModelConfigT_contra]):
    def __call__(
        self, state_dict: dict[str, object], config: ModelConfigT_contra
    ) -> HuggingFaceExport: ...


ModelT = TypeVar("ModelT", bound=Module)

ModelConfigT = TypeVar("ModelConfigT")


@final
class StandardModelFamily(ModelFamily):
    def __init__(
        self,
        name: str,
        kls: type[ModelT],
        configs: Lookup[ModelConfigT],
        factory: ModelFactory[ModelConfigT, ModelT],
        file_system: FileSystem,
        asset_download_manager: AssetDownloadManager,
        asset_config_loader: AssetConfigLoader,
        checkpoint_loader: ModelCheckpointLoader,
        supports_meta: bool,
        restrict: bool,
        sharder: ModelSharder,
        state_dict_converter: ModelStateDictConverter[ModelConfigT] | None,
        shard_specs: ShardSpecsProvider[ModelConfigT] | None,
        compiler: ModelCompiler[ModelT] | None,
        fsdp_applier: ModelFSDPApplier[ModelT] | None,
        layerwise_ac_applier: LayerwiseACApplier[ModelT] | None,
        hg_exporter: HuggingFaceExporter[ModelConfigT] | None,
        progress_reporter: ProgressReporter,
    ) -> None:
        if shard_specs is not None:
            _warn_deprecated(
                "`shard_specs` and `sharder` parameters of `StandardModelFamily` are deprecated and will be removed in fairseq2 v0.12. See src/fairseq2/sharder.py for details."
            )

        self._name = name
        self._kls: type[Module] = kls
        self._configs: Lookup[object] = configs
        self._factory: ModelFactory[Any, Module] = factory
        self._file_system = file_system
        self._asset_download_manager = asset_download_manager
        self._asset_config_loader = asset_config_loader
        self._checkpoint_loader = checkpoint_loader
        self._supports_meta = supports_meta
        self._restrict = restrict
        self._sharder = sharder
        self._state_dict_converter: ModelStateDictConverter[Any] | None = state_dict_converter  # fmt: skip
        self._shard_specs: ShardSpecsProvider[Any] | None = shard_specs
        self._compiler: ModelCompiler[Any] | None = compiler
        self._fsdp_applier: ModelFSDPApplier[Any] | None = fsdp_applier
        self._layerwise_ac_applier: LayerwiseACApplier[Any] | None = layerwise_ac_applier  # fmt: skip
        self._hg_exporter: HuggingFaceExporter[Any] | None = hg_exporter
        self._progress_reporter = progress_reporter

    @override
    def get_archs(self) -> set[str]:
        it = self._configs.iter_keys()

        return set(arch for arch in it if isinstance(arch, str))

    @override
    def maybe_get_arch_config(self, arch: str) -> object | None:
        return self._configs.maybe_get(arch)

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
                    f"Default configuration of the {self._name} model family cannot be constructed."
                ) from ex
        else:
            base_config = self.maybe_get_arch_config(arch)
            if base_config is None:
                msg = f"model_arch field of the {name} asset card is expected to be a supported model architecture, but is {arch} instead."

                raise AssetCardError(name, msg) from None

        # legacy
        base_config = self._asset_config_loader.load(
            card, base_config, config_key="model_config"
        )

        base_config = self._asset_config_loader.load(
            card, base_config, config_key="model_config_override"
        )

        return base_config

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

        path = self._asset_download_manager.download_model(uri, name)

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
            return self._do_load_model(path, config, gangs, dtype, mmap, restrict)
        except ValueError as ex:
            if has_custom_config:
                raise

            msg = f"model_config field of the {name} asset card is not a valid {self._name} model configuration."

            raise AssetCardError(name, msg) from ex
        except ModelCheckpointError as ex:
            msg = f"Model checkpoint of the {name} asset card is erroneous."

            if uri.scheme != "file":
                msg = f"{msg} Make sure that it is downloaded correctly and, if not, clean your asset cache directory at {path.parent}."

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

        return self._do_load_model(path, config, gangs, dtype, mmap, restrict)

    def _do_load_model(
        self,
        path: Path,
        config: object,
        gangs: Gangs,
        dtype: DataType,
        mmap: bool,
        restrict: bool | None,
    ) -> Module:
        raise_if_not_exists(self._file_system, path)

        model = self._do_create_model(config, gangs, dtype, self._supports_meta)

        if self._supports_meta:
            # The parameters of the model will be overwritten by the checkpoint,
            # so there is no need to redundantly initialize them.
            to_empty(model, device=gangs.root.device)

        if self._shard_specs is None:
            shard_dims = get_shard_dims(model)
        else:
            shard_dims = None

        checkpoint = self._do_iter_checkpoint(
            path, config, shard_dims, gangs, mmap, restrict
        )

        try:
            set_model_state(model, checkpoint, self._progress_reporter)
        except ModelCheckpointMismatchError as ex:
            msg = f"Checkpoint at {path} is not compatible with the {self._name} model."

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
    ) -> Iterator[tuple[str, Tensor]]:
        if not isinstance(config, self._configs.kls):
            raise TypeError(
                f"`config` must be of type `{self._configs.kls}`, but is of type `{type(config)}` instead."
            )

        if not self._supports_meta:
            _warn_deprecated(
                "In fairseq2 v0.12, `ModelFamily.iter_checkpoint` will stop supporting models that cannot be meta-initialized."
            )

        if self._shard_specs is None:
            # Initialize a dummy model solely for the purpose of extracting its
            # parameter sharding information. In the future, require this to be
            # a meta-initialization to avoid the cost of a full initialization.
            model = self._do_create_model(
                config, gangs, torch.float16, self._supports_meta
            )

            shard_dims = get_shard_dims(model)

            del model
        else:
            shard_dims = None

        return self._do_iter_checkpoint(path, config, shard_dims, gangs, mmap, restrict)

    def _do_iter_checkpoint(
        self,
        path: Path,
        config: object,
        shard_dims: Mapping[str, int] | None,
        gangs: Gangs,
        mmap: bool,
        restrict: bool | None,
    ) -> Iterator[tuple[str, Tensor]]:
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
                shard_dims=shard_dims,
            )

    def _do_create_model(
        self, config: object, gangs: Gangs, dtype: DataType, meta: bool
    ) -> Module:
        if meta:
            if not self._supports_meta:
                raise NotSupportedError(
                    f"{self._name} model family does not support meta device initialization."
                )

            device = META_DEVICE
        else:
            device = gangs.root.device

        try:
            with device, gangs:
                with default_dtype(dtype):
                    model = self._factory(config)
        except NotImplementedError as ex:
            if "'Meta' backend" not in str(ex):
                raise

            raise InternalError(
                "One or more modules of the model failed to initialize on the meta device."
            ) from ex

        if gangs.root.size != gangs.dp.size:
            if self._shard_specs is not None:
                shard_specs = self._shard_specs(config)

                try:
                    self._sharder.shard(model, gangs, shard_specs)
                except ShardSpecError as ex:
                    raise InternalError(
                        f"Shard specification of the {self._name} model family is not valid."
                    ) from ex

        return model

    @override
    def compile(self, model: Module, *args: Any, **kwargs: Any) -> None:
        if self._compiler is None:
            raise NotSupportedError(
                f"{self._name} model family does not support torch.compile()."
            )

        if not isinstance(model, self._kls):
            raise TypeError(
                f"`model` must be of type `{self._kls}`, but is of type `{type(model)}` instead."
            )

        self._compiler(model, *args, **kwargs)

    @override
    def apply_fsdp(
        self, model: Module, granularity: str, wrapper: FSDPWrapper
    ) -> Module:
        if self._fsdp_applier is None:
            raise NotSupportedError(f"{self._name} model family does not support FSDP.")

        if not isinstance(model, self._kls):
            raise TypeError(
                f"`model` must be of type `{self._kls}`, but is of type `{type(model)}` instead."
            )

        return self._fsdp_applier(model, granularity, wrapper)

    @override
    def apply_layerwise_ac(self, model: Module, every_nth_layer: int) -> Module:
        if self._layerwise_ac_applier is None:
            raise NotSupportedError(
                f"{self._name} model family does not support layerwise activation checkpointing."
            )

        if not isinstance(model, self._kls):
            raise TypeError(
                f"`model` must be of type `{self._kls}`, but is of type `{type(model)}` instead."
            )

        return self._layerwise_ac_applier(model, every_nth_layer)

    @override
    def convert_to_hugging_face(
        self, state_dict: dict[str, object], config: object
    ) -> HuggingFaceExport:
        if self._hg_exporter is None:
            raise NotSupportedError(
                f"{self._name} model family does not support Hugging Face."
            )

        if not isinstance(config, self._configs.kls):
            raise TypeError(
                f"`config` must be of type `{self._configs.kls}`, but is of type `{type(config)}` instead."
            )

        return self._hg_exporter(state_dict, config)

    @property
    @override
    def name(self) -> str:
        return self._name

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
        _warn_deprecated(
            "`ModelFamily.supports_model_parallelism` is deprecated and will be removed in fairseq2 v0.12."
        )

        return self._shard_specs is not None

    @property
    @override
    def supports_compilation(self) -> bool:
        return self._compiler is not None

    @property
    @override
    def supports_fsdp(self) -> bool:
        return self._fsdp_applier is not None

    @property
    @override
    def supports_layerwise_ac(self) -> bool:
        return self._layerwise_ac_applier is not None

    @property
    @override
    def supports_hugging_face(self) -> bool:
        return self._hg_exporter is not None
