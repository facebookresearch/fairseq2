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
from typing import Any, Final, Protocol, TypeVar, final

import torch
from torch import Tensor
from torch.nn import Module
from typing_extensions import override

from fairseq2.assets import (
    AssetCard,
    AssetCardError,
    AssetCardNotValidError,
    AssetConfigLoader,
    AssetDownloadManager,
    AssetStore,
)
from fairseq2.data_type import DataType, set_dtype
from fairseq2.device import META_DEVICE
from fairseq2.error import (
    InternalError,
    NotSupportedError,
    raise_operational_system_error,
)
from fairseq2.file_system import FileSystem, raise_if_not_exists
from fairseq2.gang import GangError, Gangs, raise_operational_gang_error, set_gangs
from fairseq2.model_checkpoint import (
    CorruptModelCheckpointError,
    ModelCheckpointLoader,
    ModelCheckpointLoadOptions,
)
from fairseq2.models.utils.checkpoint import (
    ModelCheckpointMismatchError,
    set_model_state,
)
from fairseq2.nn import get_shard_dims
from fairseq2.nn.fsdp import FSDPWrapper, load_with_sdp_gang
from fairseq2.nn.utils.module import (
    broadcast_module,
    reset_non_persistent_buffers,
    to_empty,
)
from fairseq2.runtime.dependency import DependencyLookup, get_dependency_resolver
from fairseq2.runtime.lookup import Lookup
from fairseq2.sharder import ModelSharder, ShardSpec, ShardSpecError
from fairseq2.utils.progress import ProgressReporter
from fairseq2.utils.uri import Uri
from fairseq2.utils.validation import ObjectValidator, ValidationError
from fairseq2.utils.warn import _warn_deprecated


# TODO: Will be deleted in v0.9
@dataclass
class HuggingFaceExport:
    state_dict: dict[str, object]
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
        self,
        config: object,
        gangs: Gangs,
        dtype: DataType,
        meta: bool,
        init_rank0_only: bool,
    ) -> Module: ...

    @abstractmethod
    def load_model(
        self,
        card: AssetCard,
        gangs: Gangs,
        dtype: DataType,
        config: object | None,
        load_rank0_only: bool,
        mmap: bool,
    ) -> Module: ...

    @abstractmethod
    def load_custom_model(
        self,
        path: Path,
        config: object,
        gangs: Gangs,
        dtype: DataType,
        load_rank0_only: bool,
        mmap: bool,
        restrict: bool | None,
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


class ModelGatedError(Exception):
    def __init__(self, name: str, info_url: str | None) -> None:
        super().__init__(
            f"{name} is a gated model. See {info_url} for details."
        )

        self.name = name
        self.info_url = info_url


def get_model_family_name(card: AssetCard | str) -> str:
    """
    Returns the family name of the model in the specified card.

    :raises AssetCardError: The card is erroneous and cannot be read.

    :raises AssetCardNotFoundError: ``card`` is of type ``str`` and no card with
        that name exists.

    :raises AssetCardNotValidError: The card is missing a model definition (i.e.
        `model_family` field).

    :raises ModelFamilyNotKnownError: The family of the model is not known,
        meaning has no registered :class:`ModelFamily`.
    """
    resolver = get_dependency_resolver()

    if isinstance(card, str):
        card = resolver.resolve(AssetStore).retrieve_card(card)

    families = DependencyLookup(resolver, ModelFamily)

    family = _maybe_get_model_family(card, families)
    if family is None:
        message = f"{card.name} asset card is missing a model definition (i.e. `model_family` field)."

        raise AssetCardNotValidError(card.name, message)

    return family.name


def maybe_get_model_family_name(card: AssetCard | str) -> str | None:
    """
    Returns the family name of the model in the specified card, if one is
    defined; otherwise, returns ``None``.

    :raises AssetCardError: The card is erroneous and cannot be read.

    :raises AssetCardNotFoundError: ``card`` is of type ``str`` and no card with
        that name exists.

    :raises ModelFamilyNotKnownError: The family of the model is not known,
        meaning has no registered :class:`ModelFamily`.
    """
    try:
        return get_model_family_name(card)
    except AssetCardNotValidError:
        return None


def _maybe_get_model_family(
    card: AssetCard, families: Lookup[ModelFamily]
) -> ModelFamily | None:
    field = card.maybe_get_field("model_family")
    if field is None:
        return None

    family_name = field.as_(str)

    family = families.maybe_get(family_name)
    if family is None:
        raise ModelFamilyNotKnownError(family_name)

    return family


class ModelFamilyNotKnownError(Exception):
    """Raised when a requested model family is not registered."""

    def __init__(self, name: str) -> None:
        super().__init__(f"{name} is not a known model family.")

        self.name = name


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
    _CONFIG_KEYS: Final = (
        "model_config_overrides",
        "model_config_override",
        "model_config",
    )

    def __init__(
        self,
        name: str,
        kls: type[ModelT],
        configs: Lookup[ModelConfigT],
        factory: ModelFactory[ModelConfigT, ModelT],
        file_system: FileSystem,
        validator: ObjectValidator,
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
        self._validator = validator
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

        for key in self._CONFIG_KEYS:
            config = self._asset_config_loader.load(card, base_config, config_key=key)

            if config is not base_config:
                try:
                    self._validator.validate(config)
                except ValidationError as ex:
                    msg = f"{key} field of the {name} asset card is not a valid {self._name} model configuration."

                    raise AssetCardError(name, msg) from ex

                return config

        return base_config

    @override
    def create_new_model(
        self,
        config: object,
        gangs: Gangs,
        dtype: DataType,
        meta: bool,
        init_rank0_only: bool,
    ) -> Module:
        if not isinstance(config, self._configs.kls):
            raise TypeError(
                f"`config` must be of type `{self._configs.kls}`, but is of type `{type(config)}` instead."
            )

        if meta and not self._supports_meta:
            raise NotSupportedError(
                f"{self._name} model family does not support meta device initialization."
            )

        if gangs.dp.rank == 0:
            model = self._do_create_model(config, gangs, dtype, meta)
        else:
            model = self._do_create_model(config, gangs, dtype, self._supports_meta)

        if meta:
            return model

        try:
            gangs.root.barrier()

            if not init_rank0_only and self._supports_meta:
                broadcast_module(model, gangs.dp)

                if gangs.dp.rank != 0:
                    reset_non_persistent_buffers(model)

                gangs.root.barrier()
        except GangError as ex:
            raise_operational_gang_error(ex)

        return model

    @override
    def load_model(
        self,
        card: AssetCard,
        gangs: Gangs,
        dtype: DataType,
        config: object | None,
        load_rank0_only: bool,
        mmap: bool,
    ) -> Module:
        if config is None:
            config = self.get_model_config(card)
        else:
            if not isinstance(config, self._configs.kls):
                raise TypeError(
                    f"`config` must be of type `{self._configs.kls}`, but is of type `{type(config)}` instead."
                )

        name = card.name

        uri_field = card.maybe_get_field("checkpoint")
        if uri_field is None:
            uri_field = card.maybe_get_field("url")
            if uri_field is not None:
                url = uri_field.as_(str)
            else:
                url = None

            raise ModelGatedError(name, url)

        uri = uri_field.as_uri()

        if uri.scheme not in self._asset_download_manager.supported_schemes:
            msg = f"checkpoint URI scheme of the {name} asset card is expected to be a supported scheme, but is {uri.scheme} instead."

            raise AssetCardError(name, msg)

        cached_path = self._download_model(uri, gangs)

        sub_path_field = card.maybe_get_field("checkpoint_path")
        if sub_path_field is not None:
            sub_pathname = sub_path_field.as_(str)

            path = cached_path.joinpath(sub_pathname)

            try:
                path = self._file_system.resolve(path)
            except OSError as ex:
                raise_operational_system_error(ex)

            if not path.is_relative_to(cached_path):
                msg = f"checkpoint_path field of the {name} asset card points to a path that is not relative to the download directory."

                raise AssetCardError(name, msg)
        else:
            path = cached_path

        # Handle legacy paths with format specifiers.
        if "shard_idx" in path.name:
            path = path.parent

        restrict_field = card.maybe_get_field("restrict")
        if restrict_field is not None:
            restrict = restrict_field.as_(bool)
        else:
            restrict = self._restrict

        try:
            return self._do_load_model(
                path, config, gangs, dtype, load_rank0_only, mmap, restrict
            )
        except CorruptModelCheckpointError as ex:
            msg = f"Model checkpoint of the {name} asset card is erroneous."

            if uri.scheme != "file":
                msg = f"{msg} Make sure that it is downloaded correctly and, if not, delete your cached version at {path}."

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
        load_rank0_only: bool,
        mmap: bool,
        restrict: bool | None,
    ) -> Module:
        if not isinstance(config, self._configs.kls):
            raise TypeError(
                f"`config` must be of type `{self._configs.kls}`, but is of type `{type(config)}` instead."
            )

        return self._do_load_model(
            path, config, gangs, dtype, load_rank0_only, mmap, restrict
        )

    def _do_load_model(
        self,
        path: Path,
        config: object,
        gangs: Gangs,
        dtype: DataType,
        load_rank0_only: bool,
        mmap: bool,
        restrict: bool | None,
    ) -> Module:
        raise_if_not_exists(self._file_system, path)

        model = self._do_create_model(config, gangs, dtype, self._supports_meta)

        if gangs.dp.rank == 0:
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

                raise CorruptModelCheckpointError(path, msg) from ex

            if self._supports_meta:
                # Non-persistent buffers are not included in the checkpoint, so we
                # have to explicitly initialize them.
                reset_non_persistent_buffers(model)

        gangs.root.barrier()

        if not load_rank0_only:
            broadcast_module(model, gangs.dp)

            if self._supports_meta:
                if gangs.dp.rank != 0:
                    reset_non_persistent_buffers(model)

                gangs.root.barrier()

        return model

    def _download_model(self, uri: Uri, gangs: Gangs) -> Path:
        if uri.scheme == "file":
            return uri.to_path()

        try:
            if gangs.root.rank == 0:
                self._asset_download_manager.download_model(uri)

            gangs.root.barrier()

            return self._asset_download_manager.download_model(uri, local_only=True)
        except GangError as ex:
            raise_operational_gang_error(ex)

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

        if shard_dims is None:
            if self._shard_specs is None:
                shard_dims = {}
            else:
                shard_dims = {k: v.dim for k, v in self._shard_specs(config).items()}

        load_options = ModelCheckpointLoadOptions(
            gangs=gangs,
            mmap=mmap,
            restrict=restrict,
            state_dict_converter=state_dict_converter,
        )

        with load_with_sdp_gang(gangs):  # Required for ShardedTensor
            yield from self._checkpoint_loader.lazy_load(path, shard_dims, load_options)

    def _do_create_model(
        self, config: object, gangs: Gangs, dtype: DataType, meta: bool
    ) -> Module:
        device = META_DEVICE if meta else gangs.root.device

        try:
            with device:
                with set_gangs(gangs, meta=True), set_dtype(dtype):
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
