# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any

from fairseq2.generator import Generator
from fairseq2.recipe.internal.generator import _GeneratorFactory
from fairseq2.runtime.dependency import (
    DependencyContainer,
    DependencyResolver,
    wire_object,
)


def _register_generator_factory(container: DependencyContainer) -> None:
    def create_generator_factory(resolver: DependencyResolver) -> _GeneratorFactory:
        def create_generator(**kwargs: Any) -> Generator:
            return wire_object(resolver, Generator, **kwargs)

        return wire_object(resolver, _GeneratorFactory, base_factory=create_generator)

    container.register(_GeneratorFactory, create_generator_factory)
