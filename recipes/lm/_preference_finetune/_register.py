# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.context import RuntimeContext
from fairseq2.registry import get_registry

# isort: split

from fairseq2.recipes.lm._preference_finetune._cpo import CpoFinetuneUnitHandler
from fairseq2.recipes.lm._preference_finetune._dpo import DpoFinetuneUnitHandler
from fairseq2.recipes.lm._preference_finetune._handler import POFinetuneUnitHandler
from fairseq2.recipes.lm._preference_finetune._orpo import OrpoFinetuneUnitHandler
from fairseq2.recipes.lm._preference_finetune._simpo import SimPOFinetuneUnitHandler


def register_po_finetune_units(context: RuntimeContext) -> None:
    registry = get_registry(context, POFinetuneUnitHandler)

    handler: POFinetuneUnitHandler

    # CPO
    handler = CpoFinetuneUnitHandler()

    registry.register(handler.name, handler)

    # DPO
    handler = DpoFinetuneUnitHandler(context)

    registry.register(handler.name, handler)

    # ORPO
    handler = OrpoFinetuneUnitHandler()

    registry.register(handler.name, handler)

    # SimPO
    handler = SimPOFinetuneUnitHandler()

    registry.register(handler.name, handler)
