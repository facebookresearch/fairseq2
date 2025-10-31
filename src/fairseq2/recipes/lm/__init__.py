# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.recipes.lm._instruction_finetune import (
    InstructionFinetuneConfig as InstructionFinetuneConfig,
)
from fairseq2.recipes.lm._instruction_finetune import (
    InstructionFinetuneCriterion as InstructionFinetuneCriterion,
)
from fairseq2.recipes.lm._instruction_finetune import (
    InstructionFinetuneDatasetSection as InstructionFinetuneDatasetSection,
)
from fairseq2.recipes.lm._instruction_finetune import (
    InstructionFinetuneUnit as InstructionFinetuneUnit,
)
from fairseq2.recipes.lm._instruction_finetune import (
    InstructionLossEvalUnit as InstructionLossEvalUnit,
)
from fairseq2.recipes.lm._instruction_finetune import (
    load_instruction_finetuner as load_instruction_finetuner,
)
from fairseq2.recipes.lm._instruction_finetune import (
    register_instruction_finetune_configs as register_instruction_finetune_configs,
)
from fairseq2.recipes.lm._loss_eval import (
    CausalLMLossEvalConfig as CausalLMLossEvalConfig,
)
from fairseq2.recipes.lm._loss_eval import (
    CausalLMLossEvalDatasetSection as CausalLMLossEvalDatasetSection,
)
from fairseq2.recipes.lm._loss_eval import (
    load_clm_loss_evaluator as load_clm_loss_evaluator,
)
from fairseq2.recipes.lm._loss_eval import (
    register_clm_loss_eval_configs as register_clm_loss_eval_configs,
)
from fairseq2.recipes.lm._online_finetune._generative_judge import (
    GeneralVerifierExtractorHandler as GeneralVerifierExtractorHandler,
)
from fairseq2.recipes.lm._online_finetune._generative_judge import (
    J1PairwiseScoreExtractor as J1PairwiseScoreExtractor,
)
from fairseq2.recipes.lm._online_finetune._generative_judge import (
    J1PairwiseScoreExtractorHandler as J1PairwiseScoreExtractorHandler,
)
from fairseq2.recipes.lm._online_finetune._generative_judge import (
    J1PointwiseExtractor as J1PointwiseExtractor,
)
from fairseq2.recipes.lm._online_finetune._generative_judge import (
    J1PointwiseExtractorHandler as J1PointwiseExtractorHandler,
)
from fairseq2.recipes.lm._online_finetune._generative_judge import (
    SelfAugmentingExtractor as SelfAugmentingExtractor,
)
from fairseq2.recipes.lm._online_finetune._generative_judge import (
    SelfAugmentingExtractorHandler as SelfAugmentingExtractorHandler,
)
from fairseq2.recipes.lm._online_finetune._generative_judge import (
    JudgmentExtractorHandler as JudgmentExtractorHandler,
)

# from fairseq2.recipes.lm._online_finetune._group_dpo import (
#     GroupDpoFinetuneUnitHandler as GroupDpoFinetuneUnitHandler,
# )
from fairseq2.recipes.lm._online_finetune._grpo import (
    GrpoFinetuneUnitHandler as GrpoFinetuneUnitHandler,
)
from fairseq2.recipes.lm._online_finetune._online_dpo import (
    OnlineDpoFinetuneUnitHandler as OnlineDpoFinetuneUnitHandler,
)
from fairseq2.recipes.lm._online_finetune._recipe import (
    OnlineFinetuneConfig,
)
from fairseq2.recipes.lm._online_finetune._recipe import (
    OnlineFinetuneDatasetSection as OnlineFinetuneDatasetSection,
)
from fairseq2.recipes.lm._online_finetune._recipe import (
    OnlineFinetuneUnitHandler,
    load_online_finetuner,
)

from fairseq2.recipes.lm._online_finetune._presets import (
    register_online_finetune_configs,
)

from fairseq2.recipes.lm._online_finetune._remote_model import (
    NoEnvAtheneRewardPipeline as NoEnvAtheneRewardPipeline,
)
from fairseq2.recipes.lm._online_finetune._remote_model import (
    NoEnvGeneralVerifierPipeline as NoEnvGeneralVerifierPipeline,
)
from fairseq2.recipes.lm._online_finetune._remote_model import (
    RemoteModelHandler as RemoteModelHandler,
)
from fairseq2.recipes.lm._online_finetune._rewards import (
    AtheneVerifier as AtheneVerifier,
)
from fairseq2.recipes.lm._online_finetune._rewards import (
    AtheneVerifierHandler as AtheneVerifierHandler,
)
from fairseq2.recipes.lm._online_finetune._rewards import (
    GenerativePairwiseVerifier as GenerativePairwiseVerifier,
)
from fairseq2.recipes.lm._online_finetune._rewards import (
    GenerativePairwiseVerifierHandler as GenerativePairwiseVerifierHandler,
)
from fairseq2.recipes.lm._online_finetune._rewards import (
    GenerativePointwiseVerifier as GenerativePointwiseVerifier,
)
from fairseq2.recipes.lm._online_finetune._rewards import (
    GenerativePointwiseVerifierHandler as GenerativePointwiseVerifierHandler,
)
from fairseq2.recipes.lm._online_finetune._rewards import GSM8kVerifier as GSM8kVerifier
from fairseq2.recipes.lm._online_finetune._rewards import (
    GSM8kVerifierHandler as GSM8kVerifierHandler,
)
from fairseq2.recipes.lm._online_finetune._rewards import (
    MathVerifyHandler as MathVerifyHandler,
)
from fairseq2.recipes.lm._online_finetune._rewards import (
    VLLMOutputRewardHandler as VLLMOutputRewardHandler,
)
from fairseq2.recipes.lm._preference_finetune._config import (
    POCriterionSection as POCriterionSection,
)
from fairseq2.recipes.lm._preference_finetune._config import (
    POFinetuneConfig as POFinetuneConfig,
)
from fairseq2.recipes.lm._preference_finetune._config import (
    POFinetuneDatasetSection as POFinetuneDatasetSection,
)
from fairseq2.recipes.lm._preference_finetune._cpo import (
    CPO_FINETUNE_UNIT as CPO_FINETUNE_UNIT,
)
from fairseq2.recipes.lm._preference_finetune._cpo import (
    CpoFinetuneConfig as CpoFinetuneConfig,
)
from fairseq2.recipes.lm._preference_finetune._cpo import (
    CpoFinetuneUnit as CpoFinetuneUnit,
)
from fairseq2.recipes.lm._preference_finetune._cpo import (
    CpoFinetuneUnitHandler as CpoFinetuneUnitHandler,
)
from fairseq2.recipes.lm._preference_finetune._dpo import (
    DPO_FINETUNE_UNIT as DPO_FINETUNE_UNIT,
)
from fairseq2.recipes.lm._preference_finetune._dpo import (
    DpoFinetuneConfig as DpoFinetuneConfig,
)
from fairseq2.recipes.lm._preference_finetune._dpo import (
    DpoFinetuneUnit as DpoFinetuneUnit,
)
from fairseq2.recipes.lm._preference_finetune._dpo import (
    DpoFinetuneUnitHandler as DpoFinetuneUnitHandler,
)
from fairseq2.recipes.lm._preference_finetune._handler import (
    POFinetuneUnitHandler as POFinetuneUnitHandler,
)
from fairseq2.recipes.lm._preference_finetune._handler import (
    UnknownPOFinetuneUnitError as UnknownPOFinetuneUnitError,
)
from fairseq2.recipes.lm._preference_finetune._orpo import (
    ORPO_FINETUNE_UNIT as ORPO_FINETUNE_UNIT,
)
from fairseq2.recipes.lm._preference_finetune._orpo import (
    OrpoFinetuneConfig as OrpoFinetuneConfig,
)
from fairseq2.recipes.lm._preference_finetune._orpo import (
    OrpoFinetuneUnit as OrpoFinetuneUnit,
)
from fairseq2.recipes.lm._preference_finetune._orpo import (
    OrpoFinetuneUnitHandler as OrpoFinetuneUnitHandler,
)
from fairseq2.recipes.lm._preference_finetune._recipe import (
    load_po_finetuner as load_po_finetuner,
)
from fairseq2.recipes.lm._preference_finetune._recipe import (
    register_po_finetune_configs as register_po_finetune_configs,
)
from fairseq2.recipes.lm._preference_finetune._simpo import (
    SIMPO_FINETUNE_UNIT as SIMPO_FINETUNE_UNIT,
)
from fairseq2.recipes.lm._preference_finetune._simpo import (
    SimPOFinetuneConfig as SimPOFinetuneConfig,
)
from fairseq2.recipes.lm._preference_finetune._simpo import (
    SimPOFinetuneUnit as SimPOFinetuneUnit,
)
from fairseq2.recipes.lm._preference_finetune._simpo import (
    SimPOFinetuneUnitHandler as SimPOFinetuneUnitHandler,
)
from fairseq2.recipes.lm._text_generate import TextGenerateConfig as TextGenerateConfig
from fairseq2.recipes.lm._text_generate import (
    TextGenerateDatasetSection as TextGenerateDatasetSection,
)
from fairseq2.recipes.lm._text_generate import TextGenerateUnit as TextGenerateUnit
from fairseq2.recipes.lm._text_generate import (
    load_text_generator as load_text_generator,
)
from fairseq2.recipes.lm._text_generate import (
    register_text_generate_configs as register_text_generate_configs,
)
from fairseq2.recipes.lm._train import CausalLMTrainConfig as CausalLMTrainConfig
from fairseq2.recipes.lm._train import CausalLMTrainUnit as CausalLMTrainUnit
from fairseq2.recipes.lm._train import TextDatasetSection as TextDatasetSection
from fairseq2.recipes.lm._train import load_clm_trainer as load_clm_trainer
from fairseq2.recipes.lm._train import (
    register_clm_train_configs as register_clm_train_configs,
)
