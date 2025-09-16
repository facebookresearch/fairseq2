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
from typing import final

from torch.nn import Module
from typing_extensions import override

from fairseq2.gang import Gangs
from fairseq2.nn import (
    BatchColumnShardedLinear,
    BatchLinear,
    BatchRowShardedLinear,
    ColumnShardedLinear,
    Linear,
    RowShardedLinear,
    ShardedEmbedding,
    StandardEmbedding,
    VocabShardedEmbedding,
)


@dataclass
class ShardSpec:
    dim: int
    """The sharded dimension."""

    region_boundary: bool = False
    """Whether the sharded dimension is at the boundary between
    two model parallel regions. Allows to avoid unnecessary communication."""

    disable_end_reduce: bool = False
    """Whether to disable a reduce planned at the end
    of the forward of a sharded layer.
    If ``True``, it is the responsibility of the layer's user to perform the reduce."""

    shard_children: bool = False
    """If set to ``True``, the sharder will recurse
    into the children of the sharded module."""


class ModuleSharder(ABC):
    @abstractmethod
    def shard(self, module: Module, gangs: Gangs, spec: ShardSpec) -> Module: ...

    @property
    @abstractmethod
    def supported_module_kls(self) -> type[Module]: ...


class ShardSpecError(Exception):
    pass


@final
class LinearSharder(ModuleSharder):
    @override
    def shard(self, module: Module, gangs: Gangs, spec: ShardSpec) -> Module:
        if not isinstance(module, Linear):
            raise TypeError(
                f"`module` must be of type `{Linear}`, but is of type `{type(module)}` instead."
            )

        if spec.dim == 0:
            return ColumnShardedLinear.from_linear(
                module, gangs.tp, gather_output=not spec.region_boundary
            )

        if spec.dim == 1:
            return RowShardedLinear.from_linear(
                module,
                gangs.tp,
                scatter_input=not spec.region_boundary,
                reduce_output=not spec.disable_end_reduce,
            )

        raise ShardSpecError(f"`spec.dim` must be 0 or 1, but is {spec.dim} instead.")

    @property
    @override
    def supported_module_kls(self) -> type[Module]:
        return Linear


@final
class BatchLinearSharder(ModuleSharder):
    @override
    def shard(self, module: Module, gangs: Gangs, spec: ShardSpec) -> Module:
        if not isinstance(module, BatchLinear):
            raise TypeError(
                f"`module` must be of type `{BatchLinearSharder}`, but is of type `{type(module)}` instead."
            )

        if spec.dim == 0:
            return BatchColumnShardedLinear.from_batch_linear(
                module, gangs.tp, gather_output=not spec.region_boundary
            )

        if spec.dim == 1:
            return BatchRowShardedLinear.from_batch_linear(
                module,
                gangs.tp,
                scatter_input=not spec.region_boundary,
                reduce_output=not spec.disable_end_reduce,
            )

        raise ShardSpecError(f"`spec.dim` must be 0 or 1, but is {spec.dim} instead.")

    @property
    @override
    def supported_module_kls(self) -> type[Module]:
        return BatchLinear


@final
class EmbeddingSharder(ModuleSharder):
    @override
    def shard(self, module: Module, gangs: Gangs, spec: ShardSpec) -> Module:
        if not isinstance(module, StandardEmbedding):
            raise TypeError(
                f"`module` must be of type `{StandardEmbedding}`, but is of type `{type(module)}` instead."
            )

        if spec.region_boundary:
            raise ShardSpecError("`spec.region_boundary` must be `False`.")

        if spec.dim == 0:
            return VocabShardedEmbedding.from_embedding(module, gangs.tp)

        if spec.dim == 1:
            return ShardedEmbedding.from_embedding(module, gangs.tp)

        raise ShardSpecError(f"`spec.dim` must be 0 or 1, but is {spec.dim} instead.")

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
    def __init__(self, sharders: Iterable[ModuleSharder]) -> None:
        self._sharders = {s.supported_module_kls: s for s in sharders}

    @override
    def shard(
        self, model: Module, gangs: Gangs, specs: Mapping[str, ShardSpec]
    ) -> None:
        if gangs.tp.size == 1:
            return

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
            should_recurse = True

            for pattern, spec in specs.items():
                if re.match(pattern, pathname):
                    kls = type(child)

                    sharder = self._sharders.get(kls)
                    if sharder is None:
                        raise ShardSpecError(
                            f"`specs` must match shardable modules, but the module at {pathname} matched by `specs['{pattern}']` is not shardable."
                        )

                    try:
                        sharded_child = sharder.shard(child, gangs, spec)
                    except ShardSpecError as ex:
                        raise ShardSpecError(
                            f"Shard specification of {pathname} matched by `specs['{pattern}']` is not valid. {str(ex)}"
                        ) from None

                    should_recurse = spec.shard_children

                    break

            if sharded_child is not None:
                module.register_module(name, sharded_child)

            if should_recurse:
                self._do_shard(child, gangs, specs, path)

            path.pop()
