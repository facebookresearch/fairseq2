# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from copy import copy
from dataclasses import dataclass, field
from typing import Dict, Final, List, cast, final

import ray
import torch
import torch.distributed
import torch.nn as nn
from torch import Tensor
from torch.nn import Module
from torcheval.metrics import Mean
from typing_extensions import override
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from fairseq2.recipes.lm._online_finetune._common import (
    compute_token_level_entropy,
    log_rollouts,
)

from fairseq2.context import RuntimeContext
from fairseq2.datasets.preference import PreferenceBatch
from fairseq2.datasets.prompt import PromptBatch
from fairseq2.gang import Gang, Gangs
from fairseq2.logging import log
from fairseq2.models.decoder import DecoderModel
from fairseq2.models.sequence import (
    SequenceBatch,
    SequenceModelOutput,
    as_auto_regressive_input,
)
from fairseq2.nn.data_parallel._fsdp import (
    fsdp_summon_full_parameters as fsdp_summon_full_parameters,
)
from fairseq2.nn.utils.module import freeze_parameters
from fairseq2.recipes.common import setup_reference_model
from fairseq2.recipes.common._distributed import broadcast_model
from fairseq2.recipes.config import (
    ReferenceModelSection,
    TrainerSection,
    get_config_section,
)
from fairseq2.recipes.lm._online_finetune._common import (
    GRPOBatch,
    OnlineCriterionSection,
    collate_with_target_mask,
    combine_prompts_responses_for_scoring,
    convert_vllm_output_to_ref_score,
    copy_state,
    generate_rollouts,
    prepare_grpo_batch,
)
from fairseq2.recipes.lm._online_finetune._handler import OnlineFinetuneUnitHandler
from fairseq2.recipes.lm._online_finetune._remote_vllm import (
    RemoteVllmModel,
    RemoteVllmModelHandler,
)
from fairseq2.recipes.lm._online_finetune._rewards import (
    RewardSection,
    VLLMOutputReward,
    VLLMOutputRewardHandler,
)
from fairseq2.recipes.metrics import SequenceMetricBag
from fairseq2.recipes.model import Model
from fairseq2.recipes.trainer import TrainUnit
from fairseq2.typing import DataType
from fairseq2.utils.structured import structure
from fairseq2.utils.validation import validate


@final
class GrpoFinetuneUnit(TrainUnit[SequenceBatch]):
    """Represents the language model DPO-finetuning unit with online generations. Paper: https://arxiv.org/abs/2305.18290."""

    _reference_model: Module | RemoteVllmModel | None
    _vllm_model: RemoteVllmModel
    _vllm_actors: Dict[str, RemoteVllmModel]
    _loss_config: GrpoLossConfig
    _metric_bag: GrpoFinetuneMetricBag
    _model_update_group: PyNcclCommunicator
    _sync_vllm_model_every_n_steps: int
    _sync_ref_model_every_n_steps: int
    _reward: VLLMOutputReward
    _display_name: str
    _reference_offload: bool

    def __init__(
        self,
        model: Module,
        reference_model: Module | RemoteVllmModel,
        reference_offload: bool,
        vllm_model: RemoteVllmModel,
        vllm_actors: List[RemoteVllmModel],
        reward,
        gangs: Gangs,
        loss_config: GrpoLossConfig,
        sync_vllm_model_every_n_steps: int = 1,
        sync_ref_model_every_n_step: int = -1,
    ) -> None:
        super().__init__()
        self._model = model
        self._reference_model = reference_model
        self._loss_config = loss_config
        self._vllm_actors = vllm_actors
        self._vllm_model = vllm_model
        self._gangs = gangs
        self._sync_vllm_model_every_n_steps = sync_vllm_model_every_n_steps
        self._sync_ref_model_every_n_steps = sync_ref_model_every_n_step
        self._reward = reward
        self._reference_offload = reference_offload
        self._metric_bag = GrpoFinetuneMetricBag(gangs.dp)

        self._display_name = "GRPO"

    @property
    @override
    def display_name(self) -> str | None:
        return self._display_name

    def maybe_sync_models(self, force_sync=False):

        if (
            self._sync_vllm_model_every_n_steps > 0
            and self._step_nr % self._sync_vllm_model_every_n_steps == 0
        ) or force_sync:
            with self._model.summon_full_parameters():
                if self._gangs.root.rank == 0:
                    self._vllm_model.sync_weights_with_vllm(train_model=self._model)
                self._gangs.root.barrier()

        if (
            self._sync_ref_model_every_n_steps > 0
            and self._step_nr % self._sync_ref_model_every_n_steps == 0
        ):

            if self._reference_offload:
                with self._model.summon_full_parameters():
                    if self._gangs.root.rank == 0:
                        self._reference_model.sync_weights_with_vllm(
                            train_model=self._model
                        )
                    self._gangs.root.barrier()
            else:
                with self._model.summon_full_parameters():
                    if self._gangs.root.rank == 0:
                        # syncing with ref model
                        copy_state(self._model.module, self._reference_model.module)
                    self._gangs.root.barrier()
                    broadcast_model(self._reference_model, self._gangs)

    def validate_reward(self, prompt_batch: PromptBatch) -> tuple[Tensor, int]:
        if self._gangs.dp.rank == 0:
            policy_sampling_params = copy(self._vllm_model.sampling_params)
            policy_sampling_params.n = 1
        else:
            policy_sampling_params = None
        rollouts = generate_rollouts(
            prompt_batch.prompts,
            dp_gang=self._gangs.dp,
            vllm_model=self._vllm_model,
            sampling_params=policy_sampling_params,
        )
<<<<<<< HEAD

        self.maybe_log_rollouts(prompt_batch, rollouts, "Valid")

=======
        if self._loss_config.log_rollouts:
            log_rollouts(prompt_batch, rollouts, "Valid")
>>>>>>> 099d1aa3f260425f28cdd504cbf40b4a4fbd4952
        reward_output = self._reward.process_rollouts(rollouts, prompt_batch)
        avg_reward = torch.tensor(reward_output["rewards"]).float().mean()
        self._metric_bag.update_avg_reward(avg_reward)
        self._metric_bag.update_batch_metrics(prompt_batch)
        # returning dummy loss since trainer expects it
        return torch.tensor(0.0, device=self._gangs.dp.device), prompt_batch.batch_size

    def compute_reference_logps(self, seq_batch: SequenceBatch):
        seqs_to_score = seq_batch.seqs.tolist()
        if seq_batch.padding_mask:
            prompt_lengths = (
                (~seq_batch.target_mask)
                .logical_and(seq_batch.padding_mask.materialize())
                .sum(dim=-1)
                .cpu()
            )  # extracting actual prompt lengths
            seqs_to_score = [
                seq[:l]
                for seq, l in zip(
                    seqs_to_score, seq_batch.padding_mask.seq_lens.tolist()
                )
            ]
        else:
            prompt_lengths = (~seq_batch.target_mask).sum(dim=-1).cpu()

        scored_responses = generate_rollouts(
            seqs_to_score, dp_gang=self._gangs.dp, vllm_model=self._reference_model
        )
        ref_logps = convert_vllm_output_to_ref_score(scored_responses, self._gangs)
        ref_logps = collate_with_target_mask(
            ref_logps, prompt_lengths, device=self._gangs.dp.device
        ).seqs

        return ref_logps

    @override
    def __call__(self, prompt_batch: PromptBatch) -> tuple[Tensor, int]:

        if not self.model.module.training:
            # we are in valid mode, only compute reward and return
            dummy_loss, batch_size = self.validate_reward(prompt_batch)
            return dummy_loss, batch_size

        self.maybe_sync_models()

        rollouts = generate_rollouts(
            prompt_batch.prompts, dp_gang=self._gangs.dp, vllm_model=self._vllm_model
        )
        if self._loss_config.log_rollouts:
            log_rollouts(prompt_batch, rollouts, "Train")

        reward_output = self._reward.process_rollouts(rollouts, prompt_batch)

        grpo_batch: GRPOBatch
        grpo_batch = prepare_grpo_batch(
            prompt_batch=prompt_batch,
            reward_output=reward_output,
            gangs=self._gangs,
            num_rollout_per_forward=self._loss_config.num_rollout_per_forward,
        )

        # grpo_batch, reward_output = self._reward.prepare_grpo_batch(prompt_batch, rollouts)  # loss_zeroer is used when entire batch has no valid prefrence pair

        grpo_input_batch, grpo_target_batch = as_auto_regressive_input(
            grpo_batch.prompt_rollouts
        )

        grpo_model_output = cast(
            SequenceModelOutput, self._model.module(grpo_input_batch)
        )
        logps = self._gather_lprobs(grpo_model_output, grpo_target_batch)

        tgt_logit_entropy = compute_token_level_entropy(
            grpo_model_output.logits, grpo_target_batch.target_mask
        )  # [Batch x Rollouts, 1]

        max_entropy_regularizer = (
            -tgt_logit_entropy.sum() * self._loss_config.entropy_regularizer_scale
        )
        self.metric_bag.update_logit_entropy(tgt_logit_entropy)

        if self._reference_offload:

            ref_logps = self.compute_reference_logps(grpo_batch.prompt_rollouts)

        else:
            with torch.no_grad():
                ref_grpo_model_output = cast(
                    SequenceModelOutput, self._reference_model.module(grpo_input_batch)
                )
                ref_logps = self._gather_lprobs(
                    ref_grpo_model_output, grpo_target_batch
                )

        _grpo_objective = self._compute_grpo_objective(
            logps, ref_logps, grpo_batch.rewards, grpo_target_batch
        )

        grpo_loss = -_grpo_objective + max_entropy_regularizer

        self._metric_bag.update_grpo_loss(prompt_batch, grpo_loss)

        rollouts_lengths = []
        for prompt_rollouts in reward_output["tokens"]:
            lengths = [len(r) for r in prompt_rollouts]  # lengths per the same prompt
            rollouts_lengths.append(lengths)
        rollouts_lengths = (
            torch.tensor(rollouts_lengths, device=self._gangs.dp.device)
            .float()
            .mean(dim=1)
        )  # [Batch]
        self._metric_bag.update_rollout_lengths(rollouts_lengths)

        # self._metric_bag.update_logps(batch, chosen_logps, rejected_logps)

        self._metric_bag.update_batch_metrics(
            grpo_batch.prompt_rollouts
        )  # TODO fix, now logs only the last prompt from the batch

        avg_reward = torch.tensor(reward_output["rewards"]).float().mean()
        self._metric_bag.update_avg_reward(avg_reward)

        loss = grpo_loss

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
        logps,
        ref_logps,
        advantages: Tensor,  # outcome based only for now
        target_batch: SequenceBatch,
    ) -> tuple[Tensor, Tensor, Tensor]:

        batch_size = advantages.size(0)
        num_rollouts = advantages.size(1)
        logps = logps.view(batch_size, num_rollouts, -1)
        ref_logps = ref_logps.view(batch_size, num_rollouts, -1)

        # kl penalty
        kl = (ref_logps - logps).exp() - (ref_logps - logps) - 1.0

        per_token_scaled_advantage = (logps - logps.detach()).exp() * advantages[
            :, :, None
        ]
        # per_token_scaled_advantage = logps * advantages[:,:,None]

        per_token_loss = per_token_scaled_advantage - self._loss_config.beta * kl

        target_mask = target_batch.target_mask.view(batch_size, num_rollouts, -1)

        per_seq_loss = (
            (per_token_loss * target_mask).sum(dim=-1) / target_mask.sum(dim=-1)
        ).mean(dim=1)

        # if self._gangs.root.rank == 0:
        #     from pudb.remote import set_trace
        #     set_trace(host="submit-0", port=6899, term_size=(80*4, 24*4), reverse=True)

        # self._gangs.root.barrier()

        return per_seq_loss.sum()

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
    logit_entropy: Mean
    avg_reward: Mean

    def __init__(self, gang: Gang) -> None:
        super().__init__(gang)

        # self.register_metric("rollout_logps", Mean(device=gang.device), persistent=False)
        self.register_metric(
            "rollout_lengths", Mean(device=gang.device), persistent=False
        )
        self.register_metric("grpo_loss", Mean(device=gang.device), persistent=False)
        self.register_metric("avg_reward", Mean(device=gang.device), persistent=False)
        self.register_metric(
            "logit_entropy", Mean(device=gang.device), persistent=False
        )

    @torch.inference_mode()
    def update_logit_entropy(self, logit_entropy: Tensor):
        # logit_entropy is expected to contain token-level entropy for every sequence in the current batch
        batch_size = logit_entropy.size(0)
        self.logit_entropy.update(logit_entropy.sum() / batch_size, weight=batch_size)

    @torch.inference_mode()
    def update_grpo_loss(self, batch: PromptBatch, loss: Tensor) -> None:
        """Update the GRPO loss metric.

        :param batch:
            The batch processed by the model.
        :param loss:
            The GRPO loss of ``batch``.
        """
        self.grpo_loss.update(loss / batch.batch_size, weight=batch.batch_size)

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

    @torch.inference_mode()
    def update_batch_metrics(self, batch: PreferenceBatch):
        num_examples = batch.batch_size
        self.num_examples.update(num_examples)
        if self._train:
            assert self.total_num_examples is not None
            self.total_num_examples.update(num_examples)


GRPO_FINETUNE_UNIT: Final = "grpo"


@dataclass(kw_only=True)
class GrpoLossConfig:
    num_rollout_per_forward: int = 8
    beta: float = 0.1
    """The coefficient of regularization towards the reference model."""
    entropy_regularizer_scale: float = 0.0

    log_rollouts: bool = False
    """Log rollouts during training/validation"""


@dataclass(kw_only=True)
class GrpoFinetuneConfig:
    reference_model: ReferenceModelSection | str = field(
        default_factory=lambda: ReferenceModelSection(name="fs2_llama3_1_8b_instruct")
    )
    """
    The reference model. If set to string, the recipe expects to get reference
    log-probabilities for rollouts using vllm actor.
    """

    loss_config: GrpoLossConfig = field(default_factory=lambda: GrpoLossConfig())

    reference_dtype: DataType = torch.bfloat16
    """The data type of the reference model."""

    ray_policy_actor_name: str = "vllm_policy"
    vllm_reward_model_name: str | None = None

    reward: RewardSection = field(
        default_factory=lambda: RewardSection(name="gsm8k_verifier")
    )

    sync_ref_model_every_n_steps: int = -1
    sync_vllm_model_every_n_steps: int = -1


@final
class GrpoFinetuneUnitHandler(OnlineFinetuneUnitHandler):
    _context: RuntimeContext

    def __init__(self, context: RuntimeContext) -> None:
        self._context = context

    @override
    def create(
        self, model: Module, gangs: Gangs, recipe_config: object, vllm_actors: object
    ) -> TrainUnit[PreferenceBatch]:
        criterion_section = get_config_section(
            recipe_config, "criterion", OnlineCriterionSection
        )

        config = structure(criterion_section.config, GrpoFinetuneConfig)

        validate(config)

        if isinstance(config.reference_model, ReferenceModelSection):
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
            reference_offload = False

        elif isinstance(config.reference_model, str):
            reference_model = vllm_actors[config.reference_model]
            reference_offload = True
            if config.sync_ref_model_every_n_steps != -1:
                if reference_model and reference_model.update_process_group is None:
                    raise ValueError(
                        f"Reference model actor must have update process group if we sync weights"
                    )

        else:
            raise ValueError(f"reference model {config.reference_model} not supported")

        gangs.root.barrier()

        vllm_model = vllm_actors[config.ray_policy_actor_name]

        vllm_reward_model = vllm_actors.get(config.vllm_reward_model_name, None)
        reward_registry = self._context.get_registry(VLLMOutputRewardHandler)
        reward_handler = reward_registry.get(config.reward.name)
        reward = reward_handler.create(
            reward_model=vllm_reward_model,
            reward_config=config.reward.config,
            gangs=gangs,
        )

        log.info("GRPO setup complete.")

        return GrpoFinetuneUnit(
            model,
            reference_model,
            reference_offload,
            vllm_model,
            vllm_actors,
            reward,
            gangs,
            config.loss_config,
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
