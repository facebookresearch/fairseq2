# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from typing_extensions import override

from fairseq2.data_type import DataType
from fairseq2.recipe.config import (
    CommonSection,
    GangSection,
    ReferenceModelSection,
    TokenizerSection,
)
from fairseq2.utils.validation import Validatable, ValidationResult


@dataclass
class LMEvalConfig:
    model: ReferenceModelSection = field(
        default_factory=lambda: ReferenceModelSection(name="llama3_1_8b_instruct")
    )

    tokenizer: TokenizerSection = field(
        default_factory=lambda: TokenizerSection(name="llama3_1_8b_instruct")
    )

    evaluator: LMEvaluatorSection = field(default_factory=lambda: LMEvaluatorSection())

    gang: GangSection = field(default_factory=lambda: GangSection())

    common: CommonSection = field(default_factory=lambda: CommonSection())


@dataclass
class LMEvaluatorSection(Validatable):
    tasks: list[str | dict[str, object]] = field(default_factory=lambda: ["winogrande"])

    batch_size: int = 16

    num_fewshot: int | None = None

    cache_requests: bool = False

    log_samples: bool = False

    amp: bool = False

    amp_dtype: DataType = torch.float32

    step_nr: int | None = None

    @override
    def validate(self) -> ValidationResult:
        result = ValidationResult()

        if not self.tasks:
            result.add_error("`tasks` must contain at least one task.")

        if self.batch_size < 1:
            result.add_error("`batch_size` must be greater than or equal to 1.")

        if self.num_fewshot is not None:
            if self.num_fewshot < 0:
                result.add_error("`num_fewshot` must be greater than or equal to 0.")

        return result
