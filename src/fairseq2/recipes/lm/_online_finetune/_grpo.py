# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, List, cast, final

import torch.nn as nn
import torch
import torch.distributed
from torch import Tensor
from torch.nn import Module
from torcheval.metrics import Mean
from typing_extensions import override
from fairseq2.recipes.model import Model

from fairseq2.context import RuntimeContext
from fairseq2.datasets.preference import PreferenceBatch
from fairseq2.gang import Gang, Gangs
from fairseq2.logging import log
from fairseq2.models.decoder import DecoderModel
from fairseq2.models.sequence import (
    SequenceBatch,
    SequenceModelOutput,
    as_auto_regressive_input,
)
from fairseq2.datasets.prompt import PromptBatch
from fairseq2.nn.utils.module import freeze_parameters
from fairseq2.nn.data_parallel._fsdp import (
    fsdp_summon_full_parameters as fsdp_summon_full_parameters,
)
from fairseq2.recipes.common import setup_reference_model
from fairseq2.recipes.config import (
    ReferenceModelSection,
    TrainerSection,
    get_config_section,
)
from fairseq2.recipes.lm._online_finetune._common import OnlineCriterionSection
from fairseq2.recipes.lm._online_finetune._handler import OnlineFinetuneUnitHandler
from fairseq2.recipes.trainer import TrainUnit
from fairseq2.typing import DataType
from fairseq2.utils.structured import structure
from fairseq2.utils.validation import validate
from fairseq2.recipes.common._distributed import broadcast_model

from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

import ray
from fairseq2.recipes.lm._online_finetune._rewards import VLLMOutputRewardHandler, RewardSection, VLLMOutputReward
from fairseq2.recipes.lm._online_finetune._remote_vllm import VllmConfig, RemoteVllmModelHandler, RemoteVllmModel

from fairseq2.recipes.lm._online_finetune._common import copy_state, generate_rollouts, GRPOBatch
from fairseq2.recipes.metrics import SequenceMetricBag

@final
class GrpoFinetuneUnit(TrainUnit[SequenceBatch]):
    """Represents the language model DPO-finetuning unit with online generations. Paper: https://arxiv.org/abs/2305.18290."""

    _reference_model: Module | None
    _beta: float
    _nll_scale: float
    _metric_bag: GrpoFinetuneMetricBag
    _length_normalization: bool
    _model_update_group: PyNcclCommunicator
    _sync_vllm_model_every_n_steps: int
    _sync_ref_model_every_n_steps: int
    _reward: VLLMOutputReward

    def __init__(
        self,
        model: Module,
        reference_model: Module | None,
        vllm_model: RemoteVllmModel,
        reward,
        gangs: Gangs,
        beta: float = 0.1,
        nll_scale: float = 1.0,
        length_normalization: bool = False,
        sync_vllm_model_every_n_steps: int = 1,
        sync_ref_model_every_n_step: int = -1,
    ) -> None:
        super().__init__()
        self._model = model
        self._reference_model = reference_model
        self._beta = beta
        self._nll_scale = nll_scale
        self._length_normalization = length_normalization
        self._vllm_model = vllm_model
        self._gangs = gangs
        self._sync_vllm_model_every_n_steps = sync_vllm_model_every_n_steps
        self._sync_ref_model_every_n_steps = sync_ref_model_every_n_step
        self._reward = reward
        self._metric_bag = GrpoFinetuneMetricBag(gangs.dp)

    def maybe_sync_models(self):

        if self._sync_vllm_model_every_n_steps > 0 and self._step_nr % self._sync_vllm_model_every_n_steps == 0:
            with self._model.summon_full_parameters():
                if self._gangs.root.rank == 0:
                    self._vllm_model.sync_weights_with_vllm(train_model=self._model)
                self._gangs.root.barrier()

        if self._sync_ref_model_every_n_steps > 0 and self._step_nr % self._sync_ref_model_every_n_steps == 0:
            with self._model.summon_full_parameters():
                if self._gangs.root.rank == 0:
                    # syncing with ref model
                    copy_state(self._model.module, self._reference_model.module)
                self._gangs.root.barrier()
                broadcast_model(self._reference_model, self._gangs)


    @override
    def __call__(self, prompt_batch: PromptBatch) -> tuple[Tensor, int]:

        self.maybe_sync_models()

        rollouts = generate_rollouts(prompt_batch.prompts, dp_gang=self._gangs.dp, vllm_model=self._vllm_model)

        grpo_batches: List[GRPOBatch]
        grpo_batches, reward_output = self._reward.prepare_grpo_batch(prompt_batch, rollouts)  # loss_zeroer is used when entire batch has no valid prefrence pair

        grpo_objectives = []
        for grpo_batch in grpo_batches:

            grpo_input_batch, grpo_target_batch = as_auto_regressive_input(
                grpo_batch.prompt_rollouts
            )

            grpo_model_output = cast(SequenceModelOutput, self._model.module(grpo_input_batch))

            # if self._gangs.root.rank == 0:
            #     from pudb.remote import set_trace
            #     set_trace(host="submit-0", port=6899, term_size=(80*2, 24*2), reverse=True)

            # self._gangs.root.barrier()

            with torch.no_grad():
                ref_grpo_model_output = cast(
                    SequenceModelOutput, self._reference_model.module(grpo_input_batch)
                )
            
            _grpo_objective = self._compute_grpo_objective(
                grpo_model_output,
                ref_grpo_model_output,
                grpo_batch.rewards,
                grpo_target_batch
            )

            grpo_objectives.append(_grpo_objective)

            # nll_loss = grpo_model_output.compute_loss(
            #     grpo_target_batch.seqs, loss_mask=grpo_target_batch.target_mask
            # )

        grpo_loss = - sum(grpo_objectives)  # sum per micro batch losses

        self._metric_bag.update_grpo_loss(prompt_batch, grpo_loss)

        rollouts_lengths = []
        for prompt_rollouts in reward_output["tokens"]:
            lengths = [len(r) for r in prompt_rollouts]  # lengths per the same prompt
            rollouts_lengths.append(lengths)
        rollouts_lengths = torch.tensor(rollouts_lengths, device=self._gangs.dp.device).float().mean(dim=1)  # [Batch]
        self._metric_bag.update_rollout_lengths(rollouts_lengths)

        # self._metric_bag.update_logps(batch, chosen_logps, rejected_logps)

        self._metric_bag.update_batch_metrics(grpo_batch.prompt_rollouts)  # TODO fix, now logs only the last prompt from the batch

        avg_reward = torch.tensor(reward_output["rewards"]).float().mean()
        self._metric_bag.update_avg_reward(avg_reward)

        loss = (
            grpo_loss
        )

        # if self._gangs.root.rank == 0:
        #     from pudb.remote import set_trace
        #     set_trace(host="submit-0", port=6899, term_size=(80*4, 24*4), reverse=True)

        # self._gangs.root.barrier()

        return loss, prompt_batch.batch_size

    def _gather_lprobs(
        self, output: SequenceModelOutput, target: SequenceBatch
    ) -> tuple[Tensor, Tensor]:
        assert target.target_mask is not None
        logprobs = torch.log_softmax(output.logits, dim=-1)
        per_token_logps = torch.gather(logprobs, -1, target.seqs.unsqueeze(-1)).squeeze(
            -1
        )  # [Batch, 1]

        return per_token_logps

    def _compute_grpo_objective(
        self,
        grpo_model_output: SequenceModelOutput,
        grpo_ref_model_output: SequenceModelOutput,
        advantages: Tensor,  # outcome based only for now 
        target_batch: SequenceBatch
    ) -> tuple[Tensor, Tensor, Tensor]:
        
        logps = self._gather_lprobs(grpo_model_output, target_batch)
        ref_logps = self._gather_lprobs(grpo_ref_model_output, target_batch)

        # kl penalty
        kl = (ref_logps - logps).exp() - (ref_logps - logps) - 1.0

        per_token_scaled_advantage = (logps - logps.detach()).exp() * advantages[:,None]

        per_token_loss = per_token_scaled_advantage - self._beta * kl

        per_seq_loss = (per_token_loss * target_batch.target_mask).sum(dim=-1) / target_batch.target_mask.sum(dim=-1)

        # if self._gangs.root.rank == 0:
        #     from pudb.remote import set_trace
        #     set_trace(host="submit-0", port=6899, term_size=(80*4, 24*4), reverse=True)

        # self._gangs.root.barrier()

        return per_seq_loss.mean()

    @override
    def set_step_nr(self, step_nr: int) -> None:
        self._step_nr = step_nr

    @property
    @override
    def model(self) -> Model:
        return self._model

    @property
    @override
    def metric_bag(self) -> GrpoFinetuneMetricBag:
        return self._metric_bag


class GrpoFinetuneMetricBag(SequenceMetricBag):
    """Holds the metrics of a DPO preference finetuning task."""
    # rollout_logps: Mean
    rollout_lengths: Mean
    grpo_loss: Mean
    avg_reward: Mean

    def __init__(self, gang: Gang) -> None:
        super().__init__(gang)

        # self.register_metric("rollout_logps", Mean(device=gang.device), persistent=False)
        self.register_metric("rollout_lengths", Mean(device=gang.device), persistent=False)
        self.register_metric("grpo_loss", Mean(device=gang.device), persistent=False)
        self.register_metric("avg_reward", Mean(device=gang.device), persistent=False)

    @torch.inference_mode()
    def update_grpo_loss(self, batch: PreferenceBatch, loss: Tensor) -> None:
        """Update the GRPO loss metric.

        :param batch:
            The batch processed by the model.
        :param loss:
            The GRPO loss of ``batch``.
        """
        self.grpo_loss.update(
            loss / batch.batch_size, weight=batch.batch_size
        )

    # @torch.inference_mode()
    # def update_logps(
    #     self,
    #     batch: PromptBatch,
    #     rollout_logps: Tensor,
    # ) -> None:
    #     """Update the Chosen Sequence Log Probabilities and Rejected Sequence Log Probabilities metrics.

    #     :param batch:
    #         The batch processed by the model.
    #     :param chosen_logps:
    #         The log probabilities for each sequence in ``batch.chosen``.
    #     :param rejected_logps:
    #         The log probabilities for each sequence in ``batch.rejected``.
    #     """
    #     self.rollout_logps.update(
    #         rollout_logps.sum() / batch.batch_size, weight=batch.batch_size
    #     )

    @torch.inference_mode()
    def update_rollout_lengths(
        self,
        rollout_lengths: Tensor,
    ) -> None:
        """Update the Chosen Sequence Length and Rejected Sequence Length metrics.

        :param batch:
            The batch processed by the model.
        """
        self.rollout_lengths.update(
            rollout_lengths.mean(),
            weight=rollout_lengths.size(0),
        )

    @torch.inference_mode()
    def update_avg_reward(self, avg_reward):
        self.avg_reward.update(avg_reward, weight=1)

GRPO_FINETUNE_UNIT: Final = "grpo"


@dataclass(kw_only=True)
class GrpoFinetuneConfig:
    reference_model: ReferenceModelSection = field(
        default_factory=lambda: ReferenceModelSection(name="fs2_llama3_1_8b_instruct")
    )
    """
    The reference model. If ``None``, the recipe expects to get reference
    log-probabilities for chosen and rejected targets as float values in the
    data example (fields `reference_score_rejected` and  `reference_score_chosen`).
    """

    reference_dtype: DataType = torch.bfloat16
    """The data type of the reference model."""

    # Loss
    beta: float = 0.1
    """The coefficient of regularization towards the reference model."""

    nll_scale: float = 0.0
    """The coefficient of NLL loss added to the DPO loss."""

    length_normalization: bool = False
    """Use length normalized DPO, which uses the average log probability of a sequence as the implicit reward."""

    vllm_model: VllmConfig = field(default_factory=lambda: VllmConfig(init_update_process_group=True))

    reward: RewardSection = field(default_factory=lambda: RewardSection(name="gsm8k_verifier"))

    sync_ref_model_every_n_steps: int = -1
    sync_vllm_model_every_n_steps: int = -1


@final
class GrpoFinetuneUnitHandler(OnlineFinetuneUnitHandler):
    _context: RuntimeContext

    def __init__(self, context: RuntimeContext) -> None:
        self._context = context

    @override
    def create(
        self, model: Module, gangs: Gangs, recipe_config: object
    ) -> TrainUnit[PreferenceBatch]:
        criterion_section = get_config_section(
            recipe_config, "criterion", OnlineCriterionSection
        )

        config = structure(criterion_section.config, GrpoFinetuneConfig)

        validate(config)

        reward_registry = self._context.get_registry(VLLMOutputRewardHandler)
        reward_handler = reward_registry.get(config.reward.name)
        reward = reward_handler.create(recipe_config=recipe_config, gangs=gangs)

        if config.reference_model is not None:
            log.info("Setting up GRPO with reference model.")

            trainer_section = get_config_section(
                recipe_config, "trainer", TrainerSection
            )

            reference_model = setup_reference_model(
                DecoderModel,
                self._context,
                config.reference_model.name,
                gangs,
                config.reference_dtype,
                mp=False,
                torch_compile=trainer_section.torch_compile,
            )

            freeze_parameters(reference_model.module)

            log.info("GRPO setup complete.")
        else:
            reference_model = None

        vllm_model = RemoteVllmModelHandler().create(gangs=gangs, unit_config=config)
    
        gangs.root.barrier()

        return GrpoFinetuneUnit(
            model,
            reference_model,
            vllm_model,
            reward,
            gangs,
            config.beta,
            config.nll_scale,
            config.length_normalization,
            config.sync_vllm_model_every_n_steps,
            config.sync_ref_model_every_n_steps,
        )
    
    @property
    @override
    def name(self) -> str:
        return GRPO_FINETUNE_UNIT

    @property
    @override
    def config_kls(self) -> type[object]:
        return GrpoFinetuneConfig
