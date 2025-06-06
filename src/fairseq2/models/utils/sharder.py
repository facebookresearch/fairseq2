# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol, final

from torch.nn import Module
from typing_extensions import override

from fairseq2.gang import Gangs
from fairseq2.nn import (
    ColumnShardedLinear,
    Linear,
    RowShardedLinear,
    ShardedEmbedding,
    StandardEmbedding,
    VocabShardedEmbedding,
)


@dataclass
class ModuleShardSpec:
    dim: int
    region_boundary: bool = False


class ShardedModuleMapper(Protocol):
    def __call__(
        self, module: Module, gangs: Gangs, spec: ModuleShardSpec
    ) -> Module | None: ...


def map_sharded_linear(
    module: Module, gangs: Gangs, spec: ModuleShardSpec
) -> Module | None:
    if not isinstance(module, Linear):
        return None

    if gangs.tp.size == 1:
        return module

    if spec.dim == 0:
        return ColumnShardedLinear.from_linear(
            module, gangs.tp, gather_output=not spec.region_boundary
        )

    if spec.dim == 1:
        return RowShardedLinear.from_linear(
            module, gangs.tp, scatter_input=not spec.region_boundary
        )

    return None


def map_sharded_embedding(
    module: Module, gangs: Gangs, spec: ModuleShardSpec
) -> Module | None:
    if not isinstance(module, StandardEmbedding):
        return None

    if gangs.tp.size == 1:
        return module

    # TODO: handle spec.region_boundary.
    if spec.dim == 0:
        return VocabShardedEmbedding.from_embedding(module, gangs.tp)

    if spec.dim == 1:
        return ShardedEmbedding.from_embedding(module, gangs.tp)

    return None


class ModelSharder(ABC):
    @abstractmethod
    def shard(
        self, model: Module, gangs: Gangs, specs: dict[str, ModuleShardSpec]
    ) -> None: ...


@final
class StandardModelSharder(ModelSharder):
    _mappers: Iterable[ShardedModuleMapper]

    def __init__(self, mappers: Iterable[ShardedModuleMapper]) -> None:
        self._mappers = mappers

    @override
    def shard(
        self, model: Module, gangs: Gangs, specs: dict[str, ModuleShardSpec]
    ) -> None:
        self._shard_module(model, gangs, specs, path=[])

        from fairseq2.logging import log

        log.info(model)
    def _shard_module(
        self,
        module: Module,
        gangs: Gangs,
        specs: dict[str, ModuleShardSpec],
        path: list[str],
    ) -> None:
        for name, child in module.named_children():
            path.append(name)

            pathname = ".".join(path)

            sharded_child = None

            for pattern, spec in specs.items():
                if re.match(pattern, pathname):
                    sharded_child = self._replace_module(child, gangs, spec)

                    if sharded_child is None:
                        raise ValueError(
                            f"`module.{pathname}` has a shard specification, but "
                        )

                    break

            if sharded_child is not None:
                module.register_module(name, sharded_child)
            else:
                self._shard_module(child, gangs, specs, path)

            path.pop()

    def _replace_module(
        self, module: Module, gangs: Gangs, spec: ModuleShardSpec
    ) -> Module | None:
        for mapper in self._mappers:
            sharded_module = mapper(module, gangs, spec)

            if sharded_module is not None:
                return sharded_module

        return None


def create_model_sharder() -> ModelSharder:
    mappers = [map_sharded_linear, map_sharded_embedding]

    return StandardModelSharder(mappers)
