# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, Protocol, final

from fairseq2.datasets import DataReader
from fairseq2.recipe.config import CommonSection, GeneratorSection
from fairseq2.recipe.generator import BatchT, Generator, GeneratorUnit


class _GeneratorFactory(Protocol):
    def __call__(self, **kwargs: Any) -> Generator: ...


@final
class _RecipeGeneratorFactory:
    def __init__(
        self,
        section: GeneratorSection,
        common_section: CommonSection,
        inner_factory: _GeneratorFactory,
    ) -> None:
        self._section = section
        self._common_section = common_section
        self._inner_factory = inner_factory

    def create(
        self, unit: GeneratorUnit[BatchT], data_reader: DataReader[BatchT]
    ) -> Generator:
        seed = self._common_section.seed + 3

        section = self._section

        return self._inner_factory(
            unit=unit,
            data_reader=data_reader,
            amp=section.amp,
            amp_dtype=section.amp_dtype,
            seed=seed,
        )
