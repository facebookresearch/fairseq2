# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Callable
from typing import final

from fairseq2.datasets import DataReader
from fairseq2.generator import BatchT, Generator, GeneratorUnit
from fairseq2.recipe.config import CommonSection, GeneratorSection


@final
class _GeneratorFactory:
    def __init__(
        self,
        section: GeneratorSection,
        common_section: CommonSection,
        base_factory: Callable[..., Generator],
    ) -> None:
        self._section = section
        self._common_section = common_section
        self._base_factory = base_factory

    def create(
        self, unit: GeneratorUnit[BatchT], data_reader: DataReader[BatchT]
    ) -> Generator:
        seed = self._common_section.seed + 3

        section = self._section

        return self._base_factory(
            unit=unit,
            data_reader=data_reader,
            amp=section.amp,
            amp_dtype=section.amp_dtype,
            seed=seed,
        )
