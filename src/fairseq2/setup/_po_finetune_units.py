# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import ray

from fairseq2.context import RuntimeContext
from fairseq2.recipes.lm import (  # GroupDpoFinetuneUnitHandler,
    AtheneVerifierHandler,
    CpoFinetuneUnitHandler,
    DpoFinetuneUnitHandler,
    GeneralVerifierExtractorHandler,
    GenerativePairwiseVerifierHandler,
    GenerativePointwiseVerifierHandler,
    PplHandler,
    GrpoFinetuneUnitHandler,
    GSM8kVerifierHandler,
    J1PairwiseScoreExtractorHandler,
    J1PointwiseExtractorHandler,
    SelfAugmentingExtractorHandler,
    JudgmentExtractorHandler,
    MathVerifyHandler,
    NoEnvAtheneRewardPipeline,
    NoEnvGeneralVerifierPipeline,
    OnlineDpoFinetuneUnitHandler,
    OnlineFinetuneUnitHandler,
    OrpoFinetuneUnitHandler,
    POFinetuneUnitHandler,
    RemoteModelHandler,
    SimPOFinetuneUnitHandler,
    VLLMOutputRewardHandler,
)


def _register_po_finetune_units(context: RuntimeContext) -> None:
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


def _register_online_finetune_units(context: RuntimeContext) -> None:
    registry = context.get_registry(OnlineFinetuneUnitHandler)

    # finetune units

    handler: OnlineFinetuneUnitHandler

    # Online DPO
    handler = OnlineDpoFinetuneUnitHandler(context)
    registry.register(handler.name, handler)

    # # Group DPO
    # handler = GroupDpoFinetuneUnitHandler(context)
    # registry.register(handler.name, handler)

    # GRPO
    handler = GrpoFinetuneUnitHandler(context)
    registry.register(handler.name, handler)

    # reward models

    registry = context.get_registry(VLLMOutputRewardHandler)

    # GSM8kVerifier
    handler = GSM8kVerifierHandler()
    registry.register(handler.name, handler)

    # AtheneVerifier
    handler = AtheneVerifierHandler()
    registry.register(handler.name, handler)

    # MathVerify
    handler = MathVerifyHandler()
    registry.register(handler.name, handler)

    # GenerativePointwiseVerifier
    handler = GenerativePointwiseVerifierHandler()
    registry.register(handler.name, handler)

    # PplVerifier
    handler = PplHandler()
    registry.register(handler.name, handler)

    # GenerativePairwiseVerifier
    handler = GenerativePairwiseVerifierHandler()
    registry.register(handler.name, handler)

    registry = context.get_registry(RemoteModelHandler)

    # NoEnvAtheneRewardPipeline
    handler = NoEnvAtheneRewardPipeline
    registry.register(handler.name, handler)

    # NoEnvGeneralVerifierPipeline
    handler = NoEnvGeneralVerifierPipeline
    registry.register(handler.name, handler)

    # Generative judgment extractors
    registry = context.get_registry(JudgmentExtractorHandler)

    handler = J1PointwiseExtractorHandler()
    registry.register(handler.name, handler)

    handler = SelfAugmentingExtractorHandler()
    registry.register(handler.name, handler)

    handler = J1PairwiseScoreExtractorHandler()
    registry.register(handler.name, handler)

    handler = GeneralVerifierExtractorHandler()
    registry.register(handler.name, handler)
