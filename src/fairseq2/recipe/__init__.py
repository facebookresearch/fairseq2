# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.recipe.base import EvalRecipe as EvalRecipe
from fairseq2.recipe.base import GenerationRecipe as GenerationRecipe
from fairseq2.recipe.base import RecipeContext as RecipeContext
from fairseq2.recipe.base import TrainRecipe as TrainRecipe
from fairseq2.recipe.run import evaluate as evaluate
from fairseq2.recipe.run import generate as generate
from fairseq2.recipe.run import train as train
