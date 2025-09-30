# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.nn.ddp import to_ddp
from fairseq2.nn.fsdp import to_fsdp
from fairseq2.recipe.base import RecipeContext, TrainRecipe
from fairseq2.recipe.internal.data_parallel import (
    _DDPFactory,
    _DDPModelWrapper,
    _DelegatingDPModelWrapper,
    _DPModelWrapper,
    _FSDPFactory,
    _FSDPModelWrapper,
)
from fairseq2.runtime.dependency import (
    DependencyContainer,
    DependencyResolver,
    wire_object,
)


def _register_data_parallel_wrappers(container: DependencyContainer) -> None:
    # Delegating
    container.register_type(_DPModelWrapper, _DelegatingDPModelWrapper)

    # DDP
    def create_ddp_wrapper(resolver: DependencyResolver) -> _DPModelWrapper:
        train_recipe = resolver.resolve(TrainRecipe)

        context = RecipeContext(resolver)

        static_graph = train_recipe.has_static_autograd_graph(context)

        return wire_object(resolver, _DDPModelWrapper, static_graph=static_graph)

    container.register(_DPModelWrapper, create_ddp_wrapper, key="ddp")

    container.register_instance(_DDPFactory, to_ddp)  # type: ignore[arg-type]

    # FSDP
    container.register_type(_DPModelWrapper, _FSDPModelWrapper, key="fsdp")

    container.register_instance(_FSDPFactory, to_fsdp)  # type: ignore[arg-type]
