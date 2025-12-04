# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.nn.ddp import to_ddp
from fairseq2.nn.fsdp import to_fsdp
from fairseq2.recipe.base import Recipe, RecipeContext
from fairseq2.recipe.internal.data_parallel import (
    _DataParallelModelWrapper,
    _DDPModelWrapper,
    _DelegatingDPModelWrapper,
    _FSDPModelWrapper,
)
from fairseq2.runtime.dependency import (
    DependencyContainer,
    DependencyResolver,
    wire_object,
)


def _register_data_parallel_wrappers(container: DependencyContainer) -> None:
    # Delegating
    container.register_type(_DataParallelModelWrapper, _DelegatingDPModelWrapper)

    # DDP
    def create_ddp_wrapper(resolver: DependencyResolver) -> _DataParallelModelWrapper:
        train_recipe = resolver.resolve(Recipe)

        context = RecipeContext(resolver)

        static_graph = train_recipe.has_static_autograd_graph(context)

        return wire_object(
            resolver, _DDPModelWrapper, ddp_factory=to_ddp, static_graph=static_graph
        )

    container.register(_DataParallelModelWrapper, create_ddp_wrapper, key="ddp")

    # FSDP
    def create_fsdp_wrapper(resolver: DependencyResolver) -> _FSDPModelWrapper:
        return wire_object(resolver, _FSDPModelWrapper, fsdp_factory=to_fsdp)

    container.register(_DataParallelModelWrapper, create_fsdp_wrapper, key="fsdp")
