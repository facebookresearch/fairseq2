# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.recipes.mt._common import MTCriterion as MTCriterion
from fairseq2.recipes.mt._common import MTLossSection as MTLossSection
from fairseq2.recipes.mt._eval import MTBleuChrfEvalUnit as MTBleuChrfEvalUnit
from fairseq2.recipes.mt._eval import MTEvalConfig as MTEvalConfig
from fairseq2.recipes.mt._eval import MTEvalDatasetSection as MTEvalDatasetSection
from fairseq2.recipes.mt._eval import MTLossEvalUnit as MTLossEvalUnit
from fairseq2.recipes.mt._eval import load_mt_evaluator as load_mt_evaluator
from fairseq2.recipes.mt._eval import (
    register_mt_eval_configs as register_mt_eval_configs,
)
from fairseq2.recipes.mt._train import MTTrainConfig as MTTrainConfig
from fairseq2.recipes.mt._train import MTTrainDatasetSection as MTTrainDatasetSection
from fairseq2.recipes.mt._train import MTTrainUnit as MTTrainUnit
from fairseq2.recipes.mt._train import MTValidationSection as MTValidationSection
from fairseq2.recipes.mt._train import load_mt_trainer as load_mt_trainer
from fairseq2.recipes.mt._train import (
    register_mt_train_configs as register_mt_train_configs,
)
from fairseq2.recipes.mt._translate import TextTranslateConfig as TextTranslateConfig
from fairseq2.recipes.mt._translate import (
    TextTranslateDatasetSection as TextTranslateDatasetSection,
)
from fairseq2.recipes.mt._translate import load_text_translator as load_text_translator
from fairseq2.recipes.mt._translate import (
    register_text_translate_configs as register_text_translate_configs,
)
