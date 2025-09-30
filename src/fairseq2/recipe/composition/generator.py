# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any

from fairseq2.recipe.generator import Generator
from fairseq2.recipe.internal.generator import _RecipeGeneratorFactory
from fairseq2.runtime.dependency import (
    DependencyContainer,
    DependencyResolver,
    wire_object,
)


def _register_generator_factory(container: DependencyContainer) -> None:
    def create_generator_factory(
        resolver: DependencyResolver,
    ) -> _RecipeGeneratorFactory:
        def create_generator(**kwargs: Any) -> Generator:
            return wire_object(resolver, Generator, **kwargs)

        return wire_object(
            resolver, _RecipeGeneratorFactory, inner_factory=create_generator
        )

    container.register(_RecipeGeneratorFactory, create_generator_factory)
