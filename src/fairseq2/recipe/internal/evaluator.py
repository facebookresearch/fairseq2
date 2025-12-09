# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import final

from fairseq2.datasets import DataReader
from fairseq2.evaluator import BatchT, Evaluator, EvalUnit
from fairseq2.recipe.config import CommonSection, EvaluatorSection


@final
class _EvaluatorFactory:
    def __init__(
        self,
        section: EvaluatorSection,
        common_section: CommonSection,
        base_factory: Callable[..., Evaluator],
    ) -> None:
        self._section = section
        self._common_section = common_section
        self._base_factory = base_factory

    def create(
        self,
        units: Sequence[EvalUnit[BatchT]],
        data_readers: Sequence[DataReader[BatchT]],
    ) -> Evaluator:
        seed = self._common_section.seed + 3

        section = self._section

        return self._base_factory(
            units=units,
            data_readers=data_readers,
            amp=section.amp,
            amp_dtype=section.amp_dtype,
            seed=seed,
        )
