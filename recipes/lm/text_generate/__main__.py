# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.recipe.cli import generate_main

from .recipe import TextGenRecipe

recipe = TextGenRecipe()

generate_main(recipe)
