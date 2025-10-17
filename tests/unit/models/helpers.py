# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch import Tensor
from torch.nn import Linear, Module
from typing_extensions import override

from fairseq2.assets import AssetCard
from fairseq2.data_type import DataType
from fairseq2.error import NotSupportedError
from fairseq2.gang import Gangs
from fairseq2.models import HuggingFaceExport, ModelFamily
from fairseq2.nn.fsdp import FSDPWrapper


@dataclass
class FooModelConfig:
    num_layers: int = 2


class FooModel(Module):
    def __init__(self) -> None:
        super().__init__()

        self.proj1 = Linear(10, 10, bias=True)
        self.proj2 = Linear(10, 10, bias=True)
        self.proj3 = Linear(10, 10, bias=True)


class FooModelFamily(ModelFamily):
    def __init__(self) -> None:
        self._archs = {
            "foo1": FooModelConfig(num_layers=1),
            "foo2": FooModelConfig(num_layers=2),
        }

    @override
    def get_archs(self) -> set[str]:
        return set(self._archs.keys())

    @override
    def maybe_get_arch_config(self, arch: str) -> object | None:
        return self._archs.get(arch)

    @override
    def get_model_config(self, card: AssetCard) -> object:
        raise NotSupportedError()

    @override
    def create_new_model(
        self, config: object, gangs: Gangs, dtype: DataType, meta: bool
    ) -> Module:
        raise NotSupportedError()

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
        raise NotSupportedError()

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
        raise NotSupportedError()

    @override
    def iter_checkpoint(
        self,
        path: Path,
        config: object,
        gangs: Gangs,
        mmap: bool,
        restrict: bool | None,
    ) -> Iterator[tuple[str, Tensor]]:
        raise NotSupportedError()

    @override
    def compile(self, model: Module, *args: Any, **kwargs: Any) -> None:
        raise NotSupportedError()

    @override
    def apply_fsdp(
        self, model: Module, granularity: str, wrapper: FSDPWrapper
    ) -> Module:
        raise NotSupportedError()

    @override
    def apply_layerwise_ac(self, model: Module, every_nth_layer: int) -> Module:
        raise NotSupportedError()

    @override
    def convert_to_hugging_face(
        self, state_dict: dict[str, object], config: object
    ) -> HuggingFaceExport:
        raise NotSupportedError()

    @property
    @override
    def name(self) -> str:
        return "foo"

    @property
    @override
    def kls(self) -> type[Module]:
        return FooModel

    @property
    @override
    def config_kls(self) -> type[object]:
        return FooModelConfig

    @property
    @override
    def supports_meta(self) -> bool:
        return False

    @property
    @override
    def supports_model_parallelism(self) -> bool:
        return False

    @property
    @override
    def supports_compilation(self) -> bool:
        return False

    @property
    @override
    def supports_fsdp(self) -> bool:
        return False

    @property
    @override
    def supports_layerwise_ac(self) -> bool:
        return False

    @property
    @override
    def supports_hugging_face(self) -> bool:
        return False
