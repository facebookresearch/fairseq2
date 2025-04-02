# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.context import RuntimeContext
from fairseq2.recipes.lm import (
    CpoFinetuneUnitHandler,
    DpoFinetuneUnitHandler,
    OrpoFinetuneUnitHandler,
    POFinetuneUnitHandler,
    SimPOFinetuneUnitHandler,
    OnlineDpoFinetuneUnitHandler,
    GrpoFinetuneUnitHandler,
    OnlineFinetuneUnitHandler,
    GSM8kVerifierHandler,
    NuminaMathVerifierHandler,
    SkyworkVerifierHandler,
    VLLMOutputRewardHandler,
)


def register_po_finetune_units(context: RuntimeContext) -> None:
    registry = context.get_registry(POFinetuneUnitHandler)

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


def register_online_finetune_units(context: RuntimeContext) -> None:
    registry = context.get_registry(OnlineFinetuneUnitHandler)

    # finetune units

    handler: OnlineFinetuneUnitHandler

    # Online DPO
    handler = OnlineDpoFinetuneUnitHandler(context)

    registry.register(handler.name, handler)

    handler = GrpoFinetuneUnitHandler(context)

    registry.register(handler.name, handler)

    # reward models

    registry = context.get_registry(VLLMOutputRewardHandler)

    # GSM8kVerifier
    handler = GSM8kVerifierHandler()
    registry.register(handler.name, handler)

    handler = SkyworkVerifierHandler()
    registry.register(handler.name, handler)

    # NuminaMathVerifier
    handler = NuminaMathVerifierHandler()
    registry.register(handler.name, handler)
