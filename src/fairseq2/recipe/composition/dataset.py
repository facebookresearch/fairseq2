# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.recipe.composition.config import register_config_section
from fairseq2.recipe.config import DatasetSection
from fairseq2.recipe.internal.dataset import _DatasetHolder, _RecipeDatasetOpener
from fairseq2.runtime.dependency import DependencyContainer, DependencyResolver


def register_dataset(container: DependencyContainer, section_name: str) -> None:
    register_config_section(container, section_name, DatasetSection, keyed=True)

    def get_dataset_holder(resolver: DependencyResolver) -> _DatasetHolder:
        section = resolver.resolve(DatasetSection, key=section_name)

        dataset_opener = resolver.resolve(_RecipeDatasetOpener)

        return dataset_opener.open(section_name, section)

    container.register(
        _DatasetHolder, get_dataset_holder, key=section_name, singleton=True
    )

    def get_dataset(resolver: DependencyResolver) -> object:
        dataset_holder = resolver.resolve(_DatasetHolder, key=section_name)

        return dataset_holder.dataset

    container.register(object, get_dataset, key=section_name, singleton=True)


def _register_datasets(container: DependencyContainer) -> None:
    container.register_type(_RecipeDatasetOpener)

    register_dataset(container, section_name="dataset")

    # Default Dataset
    def get_dataset_holder(resolver: DependencyResolver) -> _DatasetHolder:
        return resolver.resolve(_DatasetHolder, key="dataset")

    container.register(_DatasetHolder, get_dataset_holder)
