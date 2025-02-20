# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.context import RuntimeContext
from fairseq2.recipes.lm._preference_finetune._cpo import (
    register_cpo_finetune_unit,
)
from fairseq2.recipes.lm._preference_finetune._dpo import (
    register_dpo_finetune_unit,
)
from fairseq2.recipes.lm._preference_finetune._orpo import (
    register_orpo_finetune_unit,
)
from fairseq2.recipes.lm._preference_finetune._simpo import (
    register_simpo_finetune_unit,
)


def register_po_finetune_units(context: RuntimeContext) -> None:
    register_cpo_finetune_unit(context)
    register_dpo_finetune_unit(context)
    register_orpo_finetune_unit(context)
    register_simpo_finetune_unit(context)
