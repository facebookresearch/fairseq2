# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.recipe.cli import main

from .recipe import LMSFTRecipe

recipe = LMSFTRecipe()

main(recipe)
