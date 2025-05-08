# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from typing import Any, final, TextIO

import torch

from fairseq2.data.text.tokenizers import TextTokenDecoder, TextTokenizer
from fairseq2.gang import Gang
from fairseq2.metrics import Mean
from fairseq2.metrics.text import WerMetric
from fairseq2.models.asr import AsrModel, AsrModelOutput
from fairseq2.models.seq2seq import Seq2SeqBatch
from fairseq2.models.sequence import SequenceBatch
from fairseq2.recipes import BaseMetricBag, Model, UnitError
from fairseq2.recipes.asr import AsrCriterion, AsrScorer
from torch import Tensor
from typing_extensions import override


@final
class DoReMiAsrCriterion(AsrCriterion):
    _model: Model
    _scorer: AsrScorer | None

    def __init__(self, model: Model, scorer: AsrScorer | None = None) -> None:
        if not isinstance(model.base_module, AsrModel):
            raise TypeError(
                f"`model.base_module` must be of type `{AsrModel}`, but is of type `{type(model.base_module)}` instead."
            )

        self._model = model

        self._scorer = scorer
