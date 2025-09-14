# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.recipe.model import RecipeModel, _StandardRecipeModel
from tests.unit.models.helpers import FooModel, FooModelConfig, FooModelFamily


def create_foo_model() -> RecipeModel:
    module = FooModel()

    config = FooModelConfig()

    family = FooModelFamily()

    return _StandardRecipeModel(module, config, family)
