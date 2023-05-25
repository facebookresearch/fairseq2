# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import Any, Callable, Dict, Generic, Optional, TypeVar

import torch
from torch.nn import Module
from typing_extensions import TypeAlias

from fairseq2.assets import AssetDownloadManager, AssetStore
from fairseq2.models.utils.arch import ArchitectureRegistry
from fairseq2.models.utils.checkpoint import MapLocation, load_checkpoint

ModelT = TypeVar("ModelT", bound=Module)

ModelConfigT = TypeVar("ModelConfigT")

ModelFactory: TypeAlias = Callable[
    [ModelConfigT, Optional[torch.device], Optional[torch.dtype]], ModelT
]


# TODO: Clean up and document once asset APIs are finalized.
class ModelLoader(Generic[ModelT, ModelConfigT]):
    def __init__(
        self,
        asset_store: AssetStore,
        download_manager: AssetDownloadManager,
        model_factory: ModelFactory[ModelConfigT, ModelT],
        archs: ArchitectureRegistry[ModelConfigT],
    ) -> None:
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
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> ModelT:
        card = self.asset_store.retrieve_card(model_name)

        card.field("model_type").check_equals(self.archs.model_type)

        arch_names = self.archs.names()

        arch_name = card.field("model_arch").as_one_of(arch_names)

        config = self.archs.get_config(arch_name)

        model = self.model_factory(config, device, dtype)

        uri = card.field("checkpoint").as_uri()

        pathname = self.download_manager.download_checkpoint(
            uri, card.name, force=force, progress=progress
        )

        checkpoint = load_checkpoint(
            pathname,
            card.name,
            map_location=map_location,
            upgrader=partial(self._upgrade_checkpoint, config=config),
        )

        model.load_state_dict(checkpoint["model"])

        return model

    def _upgrade_checkpoint(
        self, checkpoint: Dict[str, Any], config: ModelConfigT
    ) -> Dict[str, Any]:
        return checkpoint
