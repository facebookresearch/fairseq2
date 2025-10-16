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
from fairseq2.recipe.base import RecipeContext, TrainRecipe
from fairseq2.recipe.internal.model import (
    _DelegatingTrainModelPreparer,
    _StandardTrainModelBootstrapper,
    _StandardTrainModelMetadataSaver,
    _StandardTrainModelPreparer,
    _TrainModelBootstrapper,
    _TrainModelMetadataSaver,
    _TrainModelPreparer,
    _TrainModelProvider,
)
from fairseq2.recipe.model import RecipeModel
from fairseq2.runtime.dependency import DependencyContainer, DependencyResolver


def _register_train_model(container: DependencyContainer) -> None:
    def get_model(resolver: DependencyResolver) -> RecipeModel:
        model_provider = resolver.resolve(_TrainModelProvider)

        return model_provider.get()

    container.register(RecipeModel, get_model, singleton=True)

    container.register_type(_TrainModelProvider)

    container.register_type(_TrainModelBootstrapper, _StandardTrainModelBootstrapper)

    container.register_type(_TrainModelMetadataSaver, _StandardTrainModelMetadataSaver)

    container.register_type(_TrainModelPreparer, _DelegatingTrainModelPreparer)

    container.collection.register_type(_TrainModelPreparer, _CustomTrainModelPreparer)
    container.collection.register_type(_TrainModelPreparer, _StandardTrainModelPreparer)


@final
class _CustomTrainModelPreparer(_TrainModelPreparer):
    def __init__(self, recipe: TrainRecipe, resolver: DependencyResolver) -> None:
        self._recipe = recipe
        self._resolver = resolver

    @override
    def prepare(self, model: RecipeModel) -> RecipeModel:
        context = RecipeContext(self._resolver)

        try:
            return self._recipe.prepare_model(context, model)
        except OSError as ex:
            raise_operational_system_error(ex)
        except GangError as ex:
            raise_operational_gang_error(ex)
