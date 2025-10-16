# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import final

from typing_extensions import override

from fairseq2.error import raise_operational_system_error
from fairseq2.gang import GangError, raise_operational_gang_error
from fairseq2.recipe.base import Recipe, RecipeContext
from fairseq2.recipe.composition.config import register_config_section
from fairseq2.recipe.config import ReferenceModelSection
from fairseq2.recipe.internal.eval_model import (
    _DelegatingEvalModelPreparer,
    _EvalModelBootstrapper,
    _EvalModelLoader,
    _EvalModelPreparer,
    _StandardEvalModelBootstrapper,
    _StandardEvalModelPreparer,
)
from fairseq2.recipe.model import RecipeModel
from fairseq2.runtime.dependency import DependencyContainer, DependencyResolver


def register_reference_model(container: DependencyContainer, section_name: str) -> None:
    register_config_section(container, section_name, ReferenceModelSection)

    def load_model(resolver: DependencyResolver) -> RecipeModel:
        section = resolver.resolve(ReferenceModelSection, key=section_name)

        model_loader = resolver.resolve(_EvalModelLoader)

        return model_loader.load(section_name, section)

    container.register(RecipeModel, load_model, key=section_name, singleton=True)


def _register_eval_model_loader(container: DependencyContainer) -> None:
    container.register_type(_EvalModelLoader)

    container.register_type(_EvalModelBootstrapper, _StandardEvalModelBootstrapper)

    container.register_type(_EvalModelPreparer, _DelegatingEvalModelPreparer)

    container.collection.register_type(_EvalModelPreparer, _CustomEvalModelPreparer)
    container.collection.register_type(_EvalModelPreparer, _StandardEvalModelPreparer)


@final
class _CustomEvalModelPreparer(_EvalModelPreparer):
    def __init__(self, recipe: Recipe, resolver: DependencyResolver) -> None:
        self._recipe = recipe
        self._resolver = resolver

    @override
    def prepare(
        self, model: RecipeModel, section_name: str, section: ReferenceModelSection
    ) -> RecipeModel:
        context = RecipeContext(self._resolver)

        try:
            if section_name == "model":
                return self._recipe.prepare_model(context, model)
            else:
                return self._recipe.prepare_reference_model(
                    context, model, section_name, section
                )
        except OSError as ex:
            raise_operational_system_error(ex)
        except GangError as ex:
            raise_operational_gang_error(ex)
