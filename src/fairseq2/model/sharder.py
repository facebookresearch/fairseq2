# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import chain
from typing import final

from torch.nn import Module
from typing_extensions import override

from fairseq2.error import NotSupportedError
from fairseq2.gang import Gangs
from fairseq2.nn import (
    ColumnShardedLinear,
    Linear,
    RowShardedLinear,
    ShardedEmbedding,
    StandardEmbedding,
    VocabShardedEmbedding,
)
from fairseq2.runtime.dependency import DependencyResolver


@dataclass
class ShardSpec:
    dim: int
    region_boundary: bool = False


class ModuleSharder(ABC):
    @abstractmethod
    def shard(self, module: Module, gangs: Gangs, spec: ShardSpec) -> Module: ...

    @property
    @abstractmethod
    def supported_module_kls(self) -> type[Module]: ...


@final
class LinearSharder(ModuleSharder):
    @override
    def shard(self, module: Module, gangs: Gangs, spec: ShardSpec) -> Module:
        if not isinstance(module, Linear):
            return module

        if spec.dim == 0:
            return ColumnShardedLinear.from_linear(
                module, gangs.tp, gather_output=not spec.region_boundary
            )

        if spec.dim == 1:
            return RowShardedLinear.from_linear(
                module, gangs.tp, scatter_input=not spec.region_boundary
            )

        raise ValueError(f"`spec.dim` must be 0 or 1, but is {spec.dim} instead.")

    @property
    @override
    def supported_module_kls(self) -> type[Module]:
        return Linear


@final
class EmbeddingSharder(ModuleSharder):
    @override
    def shard(self, module: Module, gangs: Gangs, spec: ShardSpec) -> Module:
        if not isinstance(module, StandardEmbedding):
            return module

        if spec.region_boundary:
            raise NotSupportedError(
                f"`{StandardEmbedding}` does not support `spec.region_boundary`."
            )

        if spec.dim == 0:
            return VocabShardedEmbedding.from_embedding(module, gangs.tp)

        if spec.dim == 1:
            return ShardedEmbedding.from_embedding(module, gangs.tp)

        raise ValueError(f"`spec.dim` must be 0 or 1, but is {spec.dim} instead.")

    @property
    @override
    def supported_module_kls(self) -> type[Module]:
        return StandardEmbedding


class ModelSharder(ABC):
    @abstractmethod
    def shard(
        self, model: Module, gangs: Gangs, specs: Mapping[str, ShardSpec]
    ) -> None: ...


@final
class StandardModelSharder(ModelSharder):
    _sharders: dict[type[Module], ModuleSharder]

    def __init__(self, sharders: Iterable[ModuleSharder]) -> None:
        self._sharders = {s.supported_module_kls: s for s in sharders}

    @override
    def shard(
        self, model: Module, gangs: Gangs, specs: Mapping[str, ShardSpec]
    ) -> None:
        if gangs.tp.size > 1:
            self._do_shard(model, gangs, specs, path=[])

    def _do_shard(
        self,
        module: Module,
        gangs: Gangs,
        specs: Mapping[str, ShardSpec],
        path: list[str],
    ) -> None:
        for name, child in module.named_children():
            path.append(name)

            pathname = ".".join(path)

            sharded_child = None

            for pattern, spec in specs.items():
                if re.match(pattern, pathname):
                    kls = type(child)

                    try:
                        sharder = self._sharders[kls]
                    except KeyError:
                        raise NotSupportedError(
                            f"`{kls}` does not support sharding."
                        ) from None

                    try:
                        sharded_child = sharder.shard(child, gangs, spec)
                    except ValueError as ex:
                        raise ValueError(
                            f"`model.{pathname}` cannot be sharded. See the nested exception for details."
                        ) from ex

                    break

            if sharded_child is not None:
                module.register_module(name, sharded_child)
            else:
                self._do_shard(child, gangs, specs, path)

            path.pop()


def _create_model_sharder(resolver: DependencyResolver) -> ModelSharder:
    other_sharders = resolver.resolve_all(ModuleSharder)

    sharders = [LinearSharder(), EmbeddingSharder()]

    it = chain(sharders, other_sharders)

    return StandardModelSharder(it)
