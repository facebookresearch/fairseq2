# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass

from torch.nn import Linear, Module

from fairseq2.recipe import RecipeModel, StandardRecipeModel


def create_foo_model() -> RecipeModel:
    module = FooModel()

    config = FooModelConfig()

    return StandardRecipeModel(module, config, family_name="foo")


@dataclass
class FooModelConfig:
    num_layers: int = 2


class FooModel(Module):
    def __init__(self) -> None:
        super().__init__()

        self.proj1 = Linear(10, 10, bias=True)
        self.proj2 = Linear(10, 10, bias=True)
        self.proj3 = Linear(10, 10, bias=True)
