# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.recipes.lm._instruction_finetune import (
    LMInstructionFinetuneConfig as LMInstructionFinetuneConfig,
)
from fairseq2.recipes.lm._instruction_finetune import (
    LMInstructionFinetuneCriterion as LMInstructionFinetuneCriterion,
)
from fairseq2.recipes.lm._instruction_finetune import (
    LMInstructionFinetuneUnit as LMInstructionFinetuneUnit,
)
from fairseq2.recipes.lm._instruction_finetune import (
    LMInstructionValidUnit as LMInstructionValidUnit,
)
from fairseq2.recipes.lm._instruction_finetune import (
    load_lm_instruction_finetuner as load_lm_instruction_finetuner,
)
from fairseq2.recipes.lm._instruction_finetune import (
    register_lm_instruction_finetune_configs as register_lm_instruction_finetune_configs,
)
from fairseq2.recipes.lm._nll_eval import LMNllEvalConfig as LMNllEvalConfig
from fairseq2.recipes.lm._nll_eval import load_lm_nll_evaluator as load_lm_nll_evaluator
from fairseq2.recipes.lm._nll_eval import (
    register_lm_nll_eval_configs as register_lm_nll_eval_configs,
)
from fairseq2.recipes.lm._preference_finetune._recipe import (
    LMPreferenceFinetuneConfig as LMPreferenceFinetuneConfig,
)
from fairseq2.recipes.lm._preference_finetune._recipe import (
    load_lm_preference_finetuner as load_lm_preference_finetuner,
)
from fairseq2.recipes.lm._preference_finetune._recipe import (
    register_lm_preference_finetune_configs as register_lm_preference_finetune_configs,
)
from fairseq2.recipes.lm._preference_finetune.cpo import CpoConfig as CpoConfig
from fairseq2.recipes.lm._preference_finetune.dpo import DpoConfig as DpoConfig
from fairseq2.recipes.lm._preference_finetune.orpo import OrpoConfig as OrpoConfig
from fairseq2.recipes.lm._preference_finetune.simpo import SimPOConfig as SimPOConfig
from fairseq2.recipes.lm._text_generate import (
    LMTextGenerateConfig as LMTextGenerateConfig,
)
from fairseq2.recipes.lm._text_generate import LMTextGenerateUnit as LMTextGenerateUnit
from fairseq2.recipes.lm._text_generate import (
    load_lm_text_generator as load_lm_text_generator,
)
from fairseq2.recipes.lm._text_generate import (
    register_lm_text_generate_configs as register_lm_text_generate_configs,
)

# isort: split

import fairseq2.recipes.lm._preference_finetune.cpo
import fairseq2.recipes.lm._preference_finetune.dpo
import fairseq2.recipes.lm._preference_finetune.orpo
import fairseq2.recipes.lm._preference_finetune.simpo
from fairseq2.recipes.lm._preference_finetune.utils import (
    preference_unit_factory as preference_unit_factory,
)
