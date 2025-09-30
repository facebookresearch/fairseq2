# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, final

from fairseq2.datasets import DataReader
from fairseq2.recipe.config import CommonSection, EvaluatorSection
from fairseq2.recipe.evaluator import BatchT, Evaluator, EvalUnit


class _EvaluatorFactory(Protocol):
    def __call__(self, **kwargs: Any) -> Evaluator: ...


@final
class _RecipeEvaluatorFactory:
    def __init__(
        self,
        section: EvaluatorSection,
        common_section: CommonSection,
        inner_factory: _EvaluatorFactory,
    ) -> None:
        self._section = section
        self._common_section = common_section
        self._inner_factory = inner_factory

    def create(
        self,
        units: Sequence[EvalUnit[BatchT]],
        data_readers: Sequence[DataReader[BatchT]],
    ) -> Evaluator:
        seed = self._common_section.seed + 3

        section = self._section

        return self._inner_factory(
            units=units,
            data_readers=data_readers,
            amp=section.amp,
            amp_dtype=section.amp_dtype,
            seed=seed,
        )
