# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import final

from torch.nn import Module
from typing_extensions import override

from fairseq2.error import OperationalError
from fairseq2.gang import GangError
from fairseq2.recipe.base import Recipe, RecipeContext
from fairseq2.recipe.internal.model import _ModelHolder
from fairseq2.recipe.internal.train_model import (
    _DelegatingTrainModelPreparer,
    _LastTrainModelPreparer,
    _StandardTrainModelBootstrapper,
    _StandardTrainModelMetadataSaver,
    _TrainModelBootstrapper,
    _TrainModelMetadataSaver,
    _TrainModelPreparer,
    _TrainModelProvider,
)
from fairseq2.recipe.model import RecipeModel, _StandardRecipeModel
from fairseq2.runtime.dependency import DependencyContainer, DependencyResolver


def _register_train_model(container: DependencyContainer) -> None:
    # Model
    def create_or_load_model(resolver: DependencyResolver) -> _ModelHolder:
        model_provider = resolver.resolve(_TrainModelProvider)

        return model_provider.get()

    container.register(_ModelHolder, create_or_load_model, key="model", singleton=True)

    def get_model(resolver: DependencyResolver) -> Module:
        model_holder = resolver.resolve(_ModelHolder)

        return model_holder.model

    container.register(Module, get_model, key="model", singleton=True)

    # Default Model
    def get_model_holder(resolver: DependencyResolver) -> _ModelHolder:
        return resolver.resolve(_ModelHolder, key="model")

    container.register(_ModelHolder, get_model_holder, singleton=True)

    def get_dp_model(resolver: DependencyResolver) -> Module:
        model_holder = resolver.resolve(_ModelHolder, key="model")

        return model_holder.dp_model

    container.register(Module, get_dp_model, singleton=True)

    container.register_type(_TrainModelBootstrapper, _StandardTrainModelBootstrapper)
    container.register_type(_TrainModelMetadataSaver, _StandardTrainModelMetadataSaver)
    container.register_type(_TrainModelPreparer, _DelegatingTrainModelPreparer)
    container.register_type(_TrainModelProvider)

    container.collection.register_type(_TrainModelPreparer, _UserTrainModelPreparer)
    container.collection.register_type(_TrainModelPreparer, _LastTrainModelPreparer)


@final
class _UserTrainModelPreparer(_TrainModelPreparer):
    def __init__(self, recipe: Recipe, resolver: DependencyResolver) -> None:
        self._recipe = recipe
        self._resolver = resolver

    @override
    def prepare(self, model_holder: _ModelHolder) -> None:
        context = RecipeContext(self._resolver)

        try:
            model_holder.model = self._recipe.setup_model(
                context, model_holder.model, model_holder.newly_initialized
            )

            # TODO: Deprecated, remove in v0.13
            model: RecipeModel = _StandardRecipeModel(model_holder)

            model = self._recipe.prepare_model(context, model)

            model_holder.model = model.module
        except (RuntimeError, OSError, GangError) as ex:
            raise OperationalError(
                "Failed to setup model specified in `model` section."
            ) from ex
