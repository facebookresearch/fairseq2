# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import final

from torch.nn import Module
from typing_extensions import override

from fairseq2.error import raise_operational_system_error
from fairseq2.gang import GangError, raise_operational_gang_error
from fairseq2.recipe.base import Recipe, RecipeContext
from fairseq2.recipe.composition.config import register_config_section
from fairseq2.recipe.config import ReferenceModelSection
from fairseq2.recipe.internal.model import _ModelHolder
from fairseq2.recipe.internal.reference_model import (
    _DelegatingReferenceModelPreparer,
    _LastReferenceModelPreparer,
    _ReferenceModelBootstrapper,
    _ReferenceModelLoader,
    _ReferenceModelPreparer,
    _StandardReferenceModelBootstrapper,
)
from fairseq2.recipe.model import RecipeModel, _StandardRecipeModel
from fairseq2.runtime.dependency import DependencyContainer, DependencyResolver


def register_reference_model(container: DependencyContainer, section_name: str) -> None:
    register_config_section(container, section_name, ReferenceModelSection)

    def get_model_holder(resolver: DependencyResolver) -> _ModelHolder:
        section = resolver.resolve(ReferenceModelSection, key=section_name)

        model_loader = resolver.resolve(_ReferenceModelLoader)

        return model_loader.load(section_name, section)

    container.register(_ModelHolder, get_model_holder, key=section_name, singleton=True)

    def get_model(resolver: DependencyResolver) -> Module:
        model_holder = resolver.resolve(_ModelHolder, key=section_name)

        return model_holder.model

    container.register(Module, get_model, key=section_name, singleton=True)


def _register_reference_model_loader(container: DependencyContainer) -> None:
    # fmt: off
    container.register_type(_ReferenceModelLoader)
    container.register_type(_ReferenceModelBootstrapper, _StandardReferenceModelBootstrapper)
    container.register_type(_ReferenceModelPreparer, _DelegatingReferenceModelPreparer)

    container.collection.register_type(_ReferenceModelPreparer, _UserReferenceModelPreparer)
    container.collection.register_type(_ReferenceModelPreparer, _LastReferenceModelPreparer)
    # fmt: on


@final
class _UserReferenceModelPreparer(_ReferenceModelPreparer):
    def __init__(self, recipe: Recipe, resolver: DependencyResolver) -> None:
        self._recipe = recipe
        self._resolver = resolver

    @override
    def prepare(
        self,
        model_holder: _ModelHolder,
        section_name: str,
        section: ReferenceModelSection,
    ) -> None:
        context = RecipeContext(self._resolver)

        model: RecipeModel = _StandardRecipeModel(model_holder)

        try:
            if section_name == "model":
                model = self._recipe.prepare_model(context, model)
            else:
                model = self._recipe.prepare_reference_model(
                    context, model, section_name, section
                )
        except OSError as ex:
            raise_operational_system_error(ex)
        except GangError as ex:
            raise_operational_gang_error(ex)

        model_holder.model = model.base_module
