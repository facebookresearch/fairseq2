# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import final

import torch
from torch.nn import Module
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from typing_extensions import override

from fairseq2.assets import (
    AssetCard,
    AssetCardError,
    AssetCardFieldNotFoundError,
    AssetDownloadManager,
)
from fairseq2.config_registry import ConfigNotFoundError, ConfigProvider
from fairseq2.error import ContractError, NotSupportedError
from fairseq2.gang import Gangs
from fairseq2.models._error import (
    ModelConfigLoadError,
    ModelLoadError,
    ShardedModelLoadError,
    UnknownModelArchitectureError,
    model_asset_card_error,
)
from fairseq2.nn.utils.module import (
    load_state_dict,
    reset_non_persistent_buffers,
    to_device,
    to_empty,
)
from fairseq2.typing import CPU, META, DataType
from fairseq2.utils.file import TensorLoader, TensorLoadError
from fairseq2.utils.merge import MergeError, merge_object
from fairseq2.utils.structured import StructureError, structure, unstructure
from fairseq2.utils.validation import validate


class ModelHandler(ABC):
    @abstractmethod
    def get_config(self, arch: str | None) -> object: ...

    @abstractmethod
    def load_config(self, card: AssetCard) -> object: ...

    @abstractmethod
    def create(
        self, config: object, gangs: Gangs, dtype: DataType, meta: bool
    ) -> Module: ...

    @abstractmethod
    def load(
        self, card: AssetCard, gangs: Gangs, dtype: DataType, config: object
    ) -> Module: ...

    @abstractmethod
    def load_from_path(
        self, path: Path, model_name: str, config: object, gangs: Gangs, dtype: DataType
    ) -> Module: ...

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
    def supports_sharding(self) -> bool: ...


class AbstractModelHandler(ModelHandler):
    _configs: ConfigProvider[object]
    _default_arch: str
    _asset_download_manager: AssetDownloadManager
    _tensor_loader: TensorLoader

    def __init__(
        self,
        configs: ConfigProvider[object],
        default_arch: str,
        asset_download_manager: AssetDownloadManager,
        tensor_loader: TensorLoader,
    ) -> None:
        self._configs = configs
        self._default_arch = default_arch
        self._asset_download_manager = asset_download_manager
        self._tensor_loader = tensor_loader

    @final
    @override
    def get_config(self, arch: str | None) -> object:
        if arch is None:
            effective_arch = self._default_arch
        else:
            effective_arch = arch

        try:
            return self._configs.get(effective_arch)
        except ConfigNotFoundError:
            if arch is None:
                raise ContractError(
                    f"The '{self.family}' model family does not have a configuration for the default '{self._default_arch}' architecture."
                )

            raise

    @final
    @override
    def load_config(self, card: AssetCard) -> object:
        model_name = card.name

        try:
            arch = card.field("model_arch").as_(str)
        except AssetCardFieldNotFoundError:
            arch = None
        except AssetCardError as ex:
            raise ModelConfigLoadError(
                model_name, f"The '{model_name}' asset card cannot be read. See the nested exception for details."  # fmt: skip
            ) from ex

        try:
            config = self.get_config(arch)
        except ConfigNotFoundError:
            if arch is not None:
                raise UnknownModelArchitectureError(
                    arch, self.family, model_name
                ) from None

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
                    f"The configuration class of the '{self.family}' model family cannot be unstructured. See the nested exception for details."
                ) from ex

            try:
                for config_override in reversed(config_overrides):
                    unstructured_config = merge_object(
                        unstructured_config, config_override
                    )

                config = structure(unstructured_config, type(config))
            except MergeError as ex:
                raise ModelConfigLoadError(
                    model_name, f"The '{model_name}' asset card does not contain a valid model configuration. See the nested exception for details."  # fmt: skip
                ) from ex

        return config

    @final
    @override
    def create(
        self, config: object, gangs: Gangs, dtype: DataType, meta: bool
    ) -> Module:
        if meta:
            if not self.supports_meta:
                raise NotSupportedError(
                    f"The '{self.family}' model family does not support meta device initialization."
                )

            device = META
        elif gangs.root.size != gangs.dp.size:
            device = CPU  # Avoid OOM for sharded models.
        else:
            device = gangs.root.device

        config = structure(config, self._configs.config_kls)

        validate(config)

        original_dtype = torch.get_default_dtype()

        try:
            torch.set_default_dtype(dtype)

            with device:
                model = self._create_model(config)
        except NotImplementedError as ex:
            if "'Meta' backend" not in str(ex):
                raise

            raise ContractError(
                "One or more operators in the model constructor have failed to initialize on the meta device. See the nested exception for details."
            ) from ex
        finally:
            torch.set_default_dtype(original_dtype)

        if gangs.root.size != gangs.dp.size:
            self._shard(model, config, gangs)

            if not meta and device != gangs.root.device:
                to_device(model, gangs.root.device)

        return model

    @abstractmethod
    def _create_model(self, config: object) -> Module: ...

    def _shard(self, model: Module, config: object, gangs: Gangs) -> None:
        raise NotSupportedError(
            f"The '{self.family}' model family does not support non-data parallelism."
        )

    @final
    @override
    def load(
        self, card: AssetCard, gangs: Gangs, dtype: DataType, config: object
    ) -> Module:
        model_name = card.name

        try:
            num_shards = card.field("num_shards").as_(int)
            if num_shards < 1:
                raise AssetCardError(
                    model_name, f"The value of the 'num_shards' field of the '{model_name}' asset card is expected to be a positive integer, but is {num_shards} instead."  # fmt: skip
                )
        except AssetCardFieldNotFoundError:
            num_shards = 1
        except AssetCardError as ex:
            raise model_asset_card_error(model_name) from ex

        if gangs.tp.size != num_shards:
            raise ShardedModelLoadError(model_name, num_shards, gangs.tp.size)

        if config is None:
            try:
                config = self.load_config(card)
            except ModelConfigLoadError as ex:
                raise ModelLoadError(
                    model_name, f"The '{model_name}' model configuration cannot be loaded. See the nested exception for details."  # fmt: skip
                ) from ex

            has_custom_config = False
        else:
            has_custom_config = True

        # Load the checkpoint.
        try:
            checkpoint_uri = card.field("checkpoint").as_uri()
        except AssetCardError as ex:
            raise model_asset_card_error(model_name) from ex

        shard_idx = gangs.tp.rank if num_shards > 1 else None

        path = self._asset_download_manager.download_checkpoint(
            checkpoint_uri, model_name, shard_idx=shard_idx
        )

        try:
            return self.load_from_path(path, model_name, config, gangs, dtype)
        except FileNotFoundError:
            raise ModelLoadError(
                model_name, f"The '{model_name}' model cannot be found at the '{path}' path."  # fmt: skip
            ) from None
        except ValueError as ex:
            if has_custom_config:
                raise

            raise ModelLoadError(
                model_name, f"The '{model_name}' asset card does not contain a valid model configuration of the '{self.family}' family. See the nested exception for details."  # fmt: skip
            ) from ex

    @final
    @override
    def load_from_path(
        self, path: Path, model_name: str, config: object, gangs: Gangs, dtype: DataType
    ) -> Module:
        if gangs.root.device.type == "meta":
            raise ValueError(
                "`gangs` must be on a real device, but is on the meta device instead."
            )

        try:
            checkpoint = self._tensor_loader.load(path, map_location=CPU)
        except TensorLoadError as ex:
            raise ModelLoadError(
                model_name, f"The checkpoint of the '{model_name}' model cannot be loaded. See the nested exception for details."  # fmt: skip
            ) from ex

        if "fs2" not in checkpoint:
            try:
                checkpoint = self._convert_checkpoint(checkpoint, config)
            except (KeyError, ValueError) as ex:
                raise ModelLoadError(
                    model_name, f"The checkpoint of the '{model_name}' model cannot be converted to a fairseq2 compatible format. See the nested exception for details."  # fmt: skip
                ) from ex

        # Create the model.
        model = self.create(config, gangs, dtype, meta=self.supports_meta)

        if self.supports_meta:
            # Move the model to the actual device without initializing. Its
            # state will be overwritten by the checkpoint anyways.
            to_empty(model, device=gangs.root.device)

        # Load the model state.
        model_key = checkpoint.get("model_key", "model")

        if not isinstance(model_key, str):
            raise ModelLoadError(
                model_name, f"The 'model_key' in the '{model_name}' checkpoint is expected to be of type `str`, but is of type `{type(model_key)}` instead."  # fmt: skip
            )

        try:
            state_dict = checkpoint[model_key]
        except KeyError:
            raise ModelLoadError(
                model_name, f"The '{model_name}' checkpoint does not contain a '{model_key}' key."  # fmt: skip
            ) from None

        if not isinstance(state_dict, dict):
            raise ModelLoadError(
                model_name, f"The model state dictionary in the '{model_name}' checkpoint is expected to be of type `dict`, but is of type `{type(state_dict)}` instead."  # fmt: skip
            )

        # Remove DDP 'module' prefix.
        consume_prefix_in_state_dict_if_present(state_dict, prefix="module.")

        try:
            load_state_dict(model, state_dict)
        except (KeyError, ValueError) as ex:
            raise ModelLoadError(
                model_name, f"The state of the '{model_name}' model cannot be loaded from the checkpoint. See the nested exception for details."  # fmt: skip
            ) from ex

        if self.supports_meta:
            # Non-persistent buffers are not included in the checkpoint, so we
            # have to explicitly initialize them.
            reset_non_persistent_buffers(model)

        return model

    def _convert_checkpoint(
        self, checkpoint: dict[str, object], config: object
    ) -> dict[str, object]:
        return checkpoint

    @final
    @property
    @override
    def config_kls(self) -> type[object]:
        return self._configs.config_kls

    @property
    @override
    def supports_meta(self) -> bool:
        return True

    @property
    @override
    def supports_sharding(self) -> bool:
        return False
