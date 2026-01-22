# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
As of v0.6, ``fairseq2.sharder`` module will be deprecated and will be removed
from our codebase in v0.12, which we expect to release in approximately six
months.

Based on our recent work on sequence parallelism and MoE, one thing that has
become clear is that the declarative approach offered by the ``ModelSharder``
API -originally designed to support tensor parallelism- is insufficient for
representing more complex parallelism strategies. In v0.6, we changed our
approach: we now expect parallelism strategies to be applied within model
factories. This gives model authors full control over how parallelism is applied
to their models. Migrating to this new approach is straightforward for existing
models. ``StandardMultiheadAttention``, ``GLUFeedForwardNetwork``, and
``StandardFeedForwardNetwork`` modules -which were previously pattern-matched in
``ModelSharder`` for tensor parallelism- are now "sharding-aware". This means
they accept an optional ``Gangs`` parameter and, if provided and applicable (i.e.
gangs.tp.size > 1), will shard their parameters for tensor parallelism. If you
have any custom modules with a registered ``ModuleSharder``, you should follow
the same pattern and remove their ``ModuleSharder`` from your codebase.

For reference, you can check out the Qwen model implementation.

Also note that these "sharding-aware" modules implement a new ``Sharded``
interface that exposes a single ``get_shard_dims()`` method. This interface is
now leveraged when loading model checkpoints to perform on-the-fly parameter
resharding, which was previously done based on the provided ``shard_specs``
parameter.

Once your model factory has been migrated to handle parallelism as described
above, the ``shard_spec`` argument to ``register_model_family`` as well as
your ``get_xyz_model_shard_spec`` must be removed before fairseq2 v0.12.
"""

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
    ColumnShardedLinear,
    Linear,
    RowShardedLinear,
    ShardedEmbedding,
    StandardEmbedding,
    VocabShardedEmbedding,
)
from fairseq2.utils.warn import _warn_deprecated


@dataclass
class ShardSpec:
    dim: int
    """The sharded dimension."""

    region_boundary: bool = False
    """Whether the sharded dimension is at the boundary between
    two model parallel regions. Allows to avoid unnecessary communication."""


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
                module, gangs.tp, scatter_input=not spec.region_boundary
            )

        raise ShardSpecError(f"`spec.dim` must be 0 or 1, but is {spec.dim} instead.")

    @property
    @override
    def supported_module_kls(self) -> type[Module]:
        return Linear


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
        _warn_deprecated(
            "`fairseq2.sharder` module is deprecated and will be removed in fairseq2 0.12."
        )

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

                    break

            if sharded_child is not None:
                module.register_module(name, sharded_child)
            else:
                self._do_shard(child, gangs, specs, path)

            path.pop()
