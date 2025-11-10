# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import TypeVar, final

from fairseq2.datasets import DatasetFamily

DatasetT = TypeVar("DatasetT")


@final
class RecipeDataset:
    def __init__(
        self, inner_dataset: object, config: object, family: DatasetFamily
    ) -> None:
        self._inner_dataset = inner_dataset
        self._config = config
        self._family = family

    def as_(self, kls: type[DatasetT]) -> DatasetT:
        if not isinstance(self._inner_dataset, kls):
            raise TypeError(
                f"Dataset is expected to be of type `{kls}`, but is of type `{type(self._inner_dataset)}` instead."
            )

        return self._inner_dataset

    @property
    def config(self) -> object:
        return self._config

    @property
    def family(self) -> DatasetFamily:
        return self._family
