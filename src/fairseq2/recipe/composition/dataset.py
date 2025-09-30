# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.recipe.composition.config import register_config_section
from fairseq2.recipe.config import DatasetSection
from fairseq2.recipe.dataset import RecipeDataset
from fairseq2.recipe.internal.dataset import _RecipeDatasetOpener
from fairseq2.runtime.dependency import DependencyContainer, DependencyResolver


def register_dataset(container: DependencyContainer, section_name: str) -> None:
    register_config_section(container, section_name, DatasetSection, keyed=True)

    def open_dataset(resolver: DependencyResolver) -> RecipeDataset:
        section = resolver.resolve(DatasetSection, key=section_name)

        dataset_opener = resolver.resolve(_RecipeDatasetOpener)

        return dataset_opener.open(section_name, section)

    container.register(RecipeDataset, open_dataset, key=section_name, singleton=True)


def _register_dataset(container: DependencyContainer) -> None:
    register_dataset(container, section_name="dataset")

    container.register_type(_RecipeDatasetOpener)
