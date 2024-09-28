# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.recipes.lm.preference_finetune.cpo import CpoConfig as CpoConfig
from fairseq2.recipes.lm.preference_finetune.dpo import DpoConfig as DpoConfig
from fairseq2.recipes.lm.preference_finetune.orpo import OrpoConfig as OrpoConfig
from fairseq2.recipes.lm.preference_finetune.recipe import (
    load_preference_finetuner as load_preference_finetuner,
)
from fairseq2.recipes.lm.preference_finetune.recipe import (
    preference_finetune_presets as preference_finetune_presets,
)
from fairseq2.recipes.lm.preference_finetune.simpo import SimPOConfig as SimPOConfig
from fairseq2.recipes.lm.preference_finetune.utils import (
    preference_unit_factory as preference_unit_factory,
)

# isort: split

import fairseq2.recipes.lm.preference_finetune.cpo
import fairseq2.recipes.lm.preference_finetune.dpo
import fairseq2.recipes.lm.preference_finetune.orpo
import fairseq2.recipes.lm.preference_finetune.simpo
