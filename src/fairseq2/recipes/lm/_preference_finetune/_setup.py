# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.context import RuntimeContext
from fairseq2.recipes.lm._preference_finetune._cpo import (
    CPO_FINETUNE_UNIT,
    CpoFinetuneUnitHandler,
)
from fairseq2.recipes.lm._preference_finetune._dpo import (
    DPO_FINETUNE_UNIT,
    DpoFinetuneUnitHandler,
)
from fairseq2.recipes.lm._preference_finetune._handler import (
    POFinetuneUnitHandler,
)
from fairseq2.recipes.lm._preference_finetune._orpo import (
    ORPO_FINETUNE_UNIT,
    OrpoFinetuneUnitHandler,
)
from fairseq2.recipes.lm._preference_finetune._simpo import (
    SIMPO_FINETUNE_UNIT,
    SimPOFinetuneUnitHandler,
)


def register_po_finetune_units(context: RuntimeContext) -> None:
    registry = context.get_registry(POFinetuneUnitHandler)

    registry.register(CPO_FINETUNE_UNIT, CpoFinetuneUnitHandler())
    registry.register(DPO_FINETUNE_UNIT, DpoFinetuneUnitHandler(context))
    registry.register(ORPO_FINETUNE_UNIT, OrpoFinetuneUnitHandler())
    registry.register(SIMPO_FINETUNE_UNIT, SimPOFinetuneUnitHandler())
