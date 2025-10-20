# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import TypeVar, final

from fairseq2.recipe.error import DatasetTypeNotValidError

DatasetT = TypeVar("DatasetT")


@final
class RecipeDataset:
    def __init__(
        self,
        inner_dataset: object,
        config: object,
        family_name: str,
        *,
        section_name: str = "dataset",
    ) -> None:
        self._inner_dataset = inner_dataset
        self._config = config
        self._family_name = family_name
        self._section_name = section_name

    def as_(self, kls: type[DatasetT]) -> DatasetT:
        if not isinstance(self._inner_dataset, kls):
            raise DatasetTypeNotValidError(
                type(self._inner_dataset), kls, self._section_name
            )

        return self._inner_dataset

    @property
    def config(self) -> object:
        return self._config

    @property
    def family_name(self) -> str:
        return self._family_name

    @property
    def section_name(self) -> str:
        return self._section_name
