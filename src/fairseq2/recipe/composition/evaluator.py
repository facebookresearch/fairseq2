# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any

from fairseq2.recipe.evaluator import Evaluator
from fairseq2.recipe.internal.evaluator import _RecipeEvaluatorFactory
from fairseq2.runtime.dependency import (
    DependencyContainer,
    DependencyResolver,
    wire_object,
)


def _register_evaluator_factory(container: DependencyContainer) -> None:
    def create_evaluator_factory(
        resolver: DependencyResolver,
    ) -> _RecipeEvaluatorFactory:
        def create_evaluator(**kwargs: Any) -> Evaluator:
            return wire_object(resolver, Evaluator, **kwargs)

        return wire_object(
            resolver, _RecipeEvaluatorFactory, inner_factory=create_evaluator
        )

    container.register(_RecipeEvaluatorFactory, create_evaluator_factory)
