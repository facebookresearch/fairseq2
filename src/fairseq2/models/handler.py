# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch.nn import Module
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from typing_extensions import override

from fairseq2.assets import (
    AssetCard,
    AssetCardError,
    AssetCardFieldNotFoundError,
    AssetDownloadManager,
    AssetError,
)
from fairseq2.config_registry import ConfigProvider
from fairseq2.error import ContractError, NotSupportedError
from fairseq2.gang import Gangs
from fairseq2.nn.utils.module import (
    load_state_dict,
    reset_non_persistent_buffers,
    to_device,
    to_empty,
)
from fairseq2.typing import CPU, META, DataType
from fairseq2.utils.file import TensorLoader, TensorLoadError
from fairseq2.utils.structured import (
    StructureError,
    merge_unstructured,
    structure,
    unstructure,
)


class ModelHandler(ABC):
    @abstractmethod
    def get_config(self, arch: str | None) -> object:
        ...

    @abstractmethod
    def load_config(self, card: AssetCard) -> object:
        ...

    @abstractmethod
    def create(
        self, config: object, gangs: Gangs, dtype: DataType, meta: bool
    ) -> Module:
        ...

    @abstractmethod
    def load(
        self, card: AssetCard, gangs: Gangs, dtype: DataType, config: object
    ) -> Module:
        ...

    @property
    @abstractmethod
    def family(self) -> str:
        ...

    @property
    @abstractmethod
    def kls(self) -> type[Module]:
        ...

    @property
    @abstractmethod
    def config_kls(self) -> type:
        ...

    @property
    @abstractmethod
    def supports_meta(self) -> bool:
        ...


class ModelNotFoundError(LookupError):
    name: str

    def __init__(self, name: str) -> None:
        super().__init__(f"'{name}' is not a known model.")

        self.name = name


class ModelFamilyNotFoundError(LookupError):
    name: str

    def __init__(self, name: str) -> None:
        super().__init__(f"'{name}' is not a known model family.")

        self.name = name


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

    @override
    def get_config(self, arch: str | None) -> object:
        if arch is None:
            arch = self._default_arch

        return self._configs.get(arch)

    @override
    def load_config(self, card: AssetCard) -> object:
        try:
            arch = card.field("model_arch").as_(str)
        except AssetCardFieldNotFoundError:
            arch = None

        try:
            config = self.get_config(arch)
        except LookupError:
            raise AssetError(
                card.name, f"The '{arch}' architecture of the '{card.name}' model is not a known '{self.family}' model architecture."  # fmt: skip
            ) from None

        # Override the default architecture configuration if the asset card or
        # its bases have a 'model_config' field.
        config_fields = []

        base_card: AssetCard | None = card

        while base_card is not None:
            if "model_config" in base_card.metadata:
                config_field = base_card.field("model_config").as_unstructured()

                config_fields.append(config_field)

            base_card = base_card.base

        if config_fields:
            try:
                unstructured_config = unstructure(config)
            except StructureError as ex:
                raise ContractError(
                    f"The configuration class of the '{self.family}' model family cannot be unstructured. See the nested exception for details."
                ) from ex

            try:
                for config_field in reversed(config_fields):
                    unstructured_config = merge_unstructured(
                        unstructured_config, config_field
                    )

                config = structure(unstructured_config, type(config))
            except StructureError as ex:
                raise AssetCardError(
                    card.name, f"The value of the `model_config` field of the '{card.name}' asset card cannot be parsed. See the nested exception for details."  # fmt: skip
                ) from ex

        return config

    @override
    def create(
        self, config: object, gangs: Gangs, dtype: DataType, meta: bool
    ) -> Module:
        if meta:
            if not self.supports_meta:
                raise MetaDeviceNotSupportedError(
                    "The model does not support initialization on the meta device."
                )

            device = META
        elif gangs.root.size != gangs.dp.size:
            device = CPU  # Avoid OOM for sharded models.
        else:
            device = gangs.root.device

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
    def _create_model(self, config: object) -> Module:
        ...

    def _shard(self, model: Module, config: object, gangs: Gangs) -> None:
        raise NonDataParallelismNotSupported(
            f"`gangs` has one or more non-data parallel gangs, but the '{self.family}' model family does not support non-data parallelism."
        )

    @override
    def load(
        self, card: AssetCard, gangs: Gangs, dtype: DataType, config: object
    ) -> Module:
        if gangs.root.device.type == "meta":
            raise ValueError(
                "`gangs` must be on a real device, but is on the meta device instead."
            )

        try:
            num_shards = card.field("num_shards").as_(int)
        except AssetCardFieldNotFoundError:
            num_shards = 1

        if num_shards < 1:
            raise AssetCardError(
                card.name, f"The value of the 'num_shards' field of the '{card.name}' asset card is expected to be a positive integer, but is {num_shards} instead."  # fmt: skip
            )

        tp_gang = gangs.tp  # tensor parallel

        if num_shards > 1:
            if tp_gang.size != num_shards:
                raise AssetError(
                    card.name, f"The number of processes in the tensor parallel gang is expected to match the number of checkpoint shards of the '{card.name}' model ({num_shards}), but is {tp_gang.size} instead."  # fmt: skip
                )
        else:
            if tp_gang.size != 1:
                raise AssetError(
                    card.name, f"The size of the tensor parallel gang is expected to be 1 since the checkpoint of the '{card.name}' model is not sharded, but is {tp_gang.size} instead."  # fmt: skip
                )

        if config is None:
            config = self.load_config(card)

            has_custom_config = False
        else:
            has_custom_config = True

        # Load the checkpoint.
        checkpoint_uri = card.field("checkpoint").as_uri()

        shard_idx = tp_gang.rank if num_shards > 1 else None

        path = self._asset_download_manager.download_checkpoint(
            checkpoint_uri, card.name, shard_idx=shard_idx
        )

        try:
            checkpoint = self._tensor_loader(path, map_location=CPU)
        except TensorLoadError as ex:
            raise AssetError(
                card.name, f"The checkpoint of the '{card.name}' model cannot be loaded. See the nested exception for details."  # fmt: skip
            ) from ex

        try:
            checkpoint = self._convert_checkpoint(checkpoint, config)
        except (KeyError, ValueError) as ex:
            raise AssetError(
                card.name, f"The checkpoint of the '{card.name}' model cannot be converted to a fairseq2 compatible format. See the nested exception for details."  # fmt: skip
            ) from ex

        # Make the model.
        try:
            model = self.create(config, gangs, dtype, meta=self.supports_meta)
        except ValueError as ex:
            if has_custom_config:
                raise

            raise AssetCardError(
                card.name, f"The '{card.name}' asset card does not have a valid model configuration. See the nested exception for details."  # fmt: skip
            ) from ex

        if self.supports_meta:
            # Move the model to the actual device without initializing. Its
            # state will be overwritten by the checkpoint anyways.
            to_empty(model, device=gangs.root.device)

        # Load the model state.
        model_key = checkpoint.get("model_key", "model")

        if not isinstance(model_key, str):
            raise AssetError(
                card.name, f"The 'model_key' in the '{card.name}' checkpoint is expected to be of type `str`, but is of type `{type(model_key)}` instead."  # fmt: skip
            )

        try:
            state_dict = checkpoint[model_key]
        except KeyError:
            raise AssetError(
                card.name, f"The '{card.name}' checkpoint does not contain a '{model_key}' key."  # fmt: skip
            ) from None

        if not isinstance(state_dict, dict):
            raise AssetError(
                card.name, f"The model state dictionary in the '{card.name}' checkpoint is expected to be of type `dict`, but is of type `{type(state_dict)}` instead."  # fmt: skip
            )

        # Remove DDP 'module' prefix.
        consume_prefix_in_state_dict_if_present(state_dict, prefix="module.")

        try:
            load_state_dict(model, state_dict)
        except (KeyError, ValueError) as ex:
            raise AssetError(
                card.name, f"The state of the '{card.name}' model cannot be loaded from the checkpoint. See the nested exception for details."  # fmt: skip
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

    @override
    @property
    def config_kls(self) -> type:
        return self._configs.config_kls

    @override
    @property
    def supports_meta(self) -> bool:
        return True


class MetaDeviceNotSupportedError(NotSupportedError):
    pass


class NonDataParallelismNotSupported(NotSupportedError):
    pass


def get_model_family(card: AssetCard) -> str:
    return card.field("model_family").as_(str)  # type: ignore[no-any-return]
